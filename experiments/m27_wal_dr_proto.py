"""M27 WAL-DR Step 12a: direct block reconstruction with explicit program cost.

This probe keeps the learned semantic ISA direction from WAL-E2E, but changes
the controlling objective. The main loss now optimizes direct block
reconstruction under the current Block-RVQ surface, while an explicit
program-cost term penalizes long exact programs with large correction tails.

For each target layer:
  - initialize semantic families and family-conditioned atoms on sampled block
    programs
  - train a small semantic encoder against:
      * direct block reconstruction loss
      * explicit program-cost proxy
      * secondary semantic regularizers
  - measure two surfaces:
      1. exact WAL-DR: FAMILY + low-level atoms + literal corrections
      2. dr_only: FAMILY-conditioned low-level atoms only, no corrections
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from m27_wal_e2e_proto import (
    MODEL_DIR,
    ROOT,
    TARGETS,
    SemanticISAEncoder,
    _build_current_cache,
    _build_initial_phrase_bank,
    _clone_group_with_program_matrix,
    _compare_encodings,
    _combined_family_distance,
    _eval_ids,
    _family_names_for_role,
    _flatten_group_sequences,
    _instruction_name,
    _predict_hard_from_features,
    _role_name,
    _run_dense_eval,
    _run_preencoded_eval,
    _semantic_features,
)

from dwl2_dynamic_route.src.block_vq import (
    BlockRVQEncoding,
    GroupedBlockRVQEncoding,
    _inverse_polar_transform,
    _sign_correction_matrix,
    _transform_matrix,
    _unpack_sign_bits,
)
from dwl2_dynamic_route.src.encoding_io import save_grouped_encoding_map


def _selected_row_indices(group: BlockRVQEncoding, flat_indices: torch.Tensor) -> torch.Tensor:
    return flat_indices.to(torch.int64) // int(group.stage_shape[1])


def _selected_scale_features(group: BlockRVQEncoding, flat_indices: torch.Tensor) -> torch.Tensor:
    row_idx = _selected_row_indices(group, flat_indices)
    row_scale = group.row_scale.reshape(-1)[row_idx].to(torch.float32).cpu().clamp_min(1e-8).log().unsqueeze(1)
    if group.block_scale is None:
        block_scale = torch.zeros_like(row_scale)
    else:
        block_scale = group.block_scale.reshape(-1)[flat_indices].to(torch.float32).cpu().clamp_min(1e-8).log().unsqueeze(1)
    return torch.cat([row_scale, block_scale], dim=1)


def _selected_residual_blocks(group: BlockRVQEncoding, flat_indices: torch.Tensor) -> torch.Tensor:
    if group.residual_correction == "none" or group.residual_signs is None or group.residual_scale is None:
        return torch.zeros(int(flat_indices.numel()), group.block_size, dtype=torch.float32)
    signs = _unpack_sign_bits(group.residual_signs[flat_indices], group.block_size).to(torch.float32)
    signs = signs * 2.0 - 1.0
    correction = signs * group.residual_scale.reshape(-1)[flat_indices].to(torch.float32).unsqueeze(1)
    correction = correction @ _sign_correction_matrix(group.block_size, device=correction.device)
    return correction.cpu()


def _apply_block_postprocess(
    group: BlockRVQEncoding,
    stage_sum: torch.Tensor,
    flat_indices: torch.Tensor,
    residual_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    out = stage_sum.to(torch.float32)
    if residual_blocks is not None:
        out = out + residual_blocks.to(torch.float32)
    if group.block_scale is not None:
        out = out * group.block_scale.reshape(-1)[flat_indices].to(torch.float32).unsqueeze(1)
    if group.transform_kind == "polar":
        out = _inverse_polar_transform(out)
    else:
        transform = group.transform_matrix
        if transform is None:
            transform = _transform_matrix(group.transform_kind, group.block_size, device=out.device)
        if transform is not None:
            out = out @ transform.to(torch.float32)
    if group.transform_bias is not None:
        out = out + group.transform_bias.to(torch.float32)
    row_idx = _selected_row_indices(group, flat_indices)
    out = out * group.row_scale.reshape(-1)[row_idx].to(torch.float32).unsqueeze(1)
    return out


def _reconstruct_selected_blocks(group: BlockRVQEncoding, flat_indices: torch.Tensor) -> torch.Tensor:
    stage_sum = torch.zeros(int(flat_indices.numel()), group.block_size, dtype=torch.float32)
    stage_scales = None if group.stage_scales is None else group.stage_scales.to(torch.float32)
    for stage_idx, (ids, codebook) in enumerate(zip(group.stage_ids, group.codebooks)):
        stage_ids = ids.reshape(-1)[flat_indices].to(torch.int64)
        stage = codebook[stage_ids].to(torch.float32)
        if stage_scales is not None:
            stage = stage * stage_scales[stage_idx]
        stage_sum = stage_sum + stage.cpu()
    residual_blocks = _selected_residual_blocks(group, flat_indices)
    return _apply_block_postprocess(group, stage_sum, flat_indices, residual_blocks=residual_blocks).cpu()


def _sample_training_corpus(
    enc: GroupedBlockRVQEncoding,
    max_rows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(enc.groups) != 1:
        raise ValueError("WAL-DR Step 12a currently expects a single grouped encoding per target layer")
    group = enc.groups[0]
    seq_mat = _flatten_group_sequences(group)
    total_rows = int(seq_mat.shape[0])
    if total_rows <= max_rows:
        pick = torch.arange(total_rows, dtype=torch.int64)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        pick = torch.randperm(total_rows, generator=generator)[:max_rows]
    sample_sequences = seq_mat[pick].contiguous()
    sample_blocks = _reconstruct_selected_blocks(group, pick).contiguous()
    sample_scale_features = _selected_scale_features(group, pick).contiguous()
    return sample_sequences, sample_blocks, sample_scale_features, pick.contiguous()


def _learn_initial_families(
    name: str,
    sample_sequences: torch.Tensor,
    raw_features: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    num_families = int(args.num_families)
    if int(sample_sequences.shape[0]) < num_families:
        raise ValueError("train_samples must be >= num_families")
    base = int(sample_sequences.max().item()) + 1
    feature_mean = raw_features.mean(dim=0)
    feature_std = raw_features.std(dim=0).clamp_min(1e-6)
    norm_features = ((raw_features - feature_mean) / feature_std).to(torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(int(sample_sequences.shape[0]), generator=generator)
    feature_centroids = norm_features[perm[:num_families]].clone()
    sequence_centroids = sample_sequences[perm[:num_families]].clone()
    history = []
    semantic_weight = float(args.init_semantic_weight)
    sequence_weight = float(args.init_sequence_weight)
    for iter_idx in range(int(args.init_family_iters)):
        combined, feature_dist, sequence_dist = _combined_family_distance(
            norm_features,
            sample_sequences,
            feature_centroids,
            sequence_centroids,
            semantic_weight,
            sequence_weight,
        )
        best_dist, assignments = combined.min(dim=1)
        counts = torch.bincount(assignments, minlength=num_families)
        updated_feature_centroids = feature_centroids.clone()
        updated_sequence_centroids = sequence_centroids.clone()
        for family_idx in range(num_families):
            mask = assignments == family_idx
            if bool(mask.any()):
                updated_feature_centroids[family_idx] = norm_features[mask].mean(dim=0)
                for pos in range(int(sample_sequences.shape[1])):
                    hist = torch.bincount(sample_sequences[mask, pos].to(torch.int64), minlength=base)
                    updated_sequence_centroids[family_idx, pos] = int(hist.argmax().item())
            else:
                reseed = int(torch.randint(0, int(sample_sequences.shape[0]), (1,), generator=generator).item())
                updated_feature_centroids[family_idx] = norm_features[reseed]
                updated_sequence_centroids[family_idx] = sample_sequences[reseed]
        feature_centroids = updated_feature_centroids.contiguous()
        sequence_centroids = updated_sequence_centroids.contiguous()
        history.append(
            {
                "iter": int(iter_idx + 1),
                "mean_combined": float(best_dist.mean().item()),
                "mean_feature_dist": float(feature_dist.gather(1, assignments.unsqueeze(1)).mean().item()),
                "mean_sequence_dist": float(sequence_dist.gather(1, assignments.unsqueeze(1)).mean().item()),
                "min_family_count": int(counts.min().item()),
            }
        )

    combined, feature_dist, sequence_dist = _combined_family_distance(
        norm_features,
        sample_sequences,
        feature_centroids,
        sequence_centroids,
        semantic_weight,
        sequence_weight,
    )
    _, assignments = combined.min(dim=1)
    centroid_stats = []
    for family_idx in range(num_families):
        mask = assignments == family_idx
        if bool(mask.any()):
            centroid_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "modal_match": float(raw_features[mask, 1].mean().item()),
                }
            )
        else:
            centroid_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float("inf"),
                    "modal_match": 0.0,
                }
            )
    order = sorted(centroid_stats, key=lambda item: (item["avg_surprisal"], -item["modal_match"]))
    permute = torch.tensor([int(item["family_idx"]) for item in order], dtype=torch.int64)
    inverse = torch.empty_like(permute)
    inverse[permute] = torch.arange(num_families, dtype=torch.int64)
    feature_centroids = feature_centroids[permute]
    sequence_centroids = sequence_centroids[permute]
    assignments = inverse[assignments]
    family_names = _family_names_for_role(_role_name(name), num_families)
    return {
        "base": base,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "raw_features": raw_features,
        "norm_features": norm_features,
        "feature_centroids": feature_centroids,
        "sequence_centroids": sequence_centroids,
        "sample_family_ids": assignments,
        "family_names": family_names,
        "family_history": history,
    }


def _prepare_stage_vector_bank(group: BlockRVQEncoding, base: int) -> torch.Tensor:
    raw_len = len(group.codebooks)
    bank = torch.zeros(raw_len, base, group.block_size, dtype=torch.float32)
    stage_scales = None if group.stage_scales is None else group.stage_scales.to(torch.float32)
    for stage_idx, codebook in enumerate(group.codebooks):
        stage = codebook.to(torch.float32)
        if stage_scales is not None:
            stage = stage * stage_scales[stage_idx]
        bank[stage_idx, : int(stage.shape[0])] = stage.cpu()
    return bank.contiguous()


def _hard_program_cost(program: torch.Tensor, target_sequences: torch.Tensor, phrase_len: int) -> torch.Tensor:
    num_slots = int(program.shape[1] // phrase_len)
    cost = torch.ones(int(program.shape[0]), dtype=torch.float32)
    for slot_idx in range(num_slots):
        start = slot_idx * phrase_len
        end = start + phrase_len
        dist = (program[:, start:end] != target_sequences[:, start:end]).sum(dim=1).to(torch.float32)
        cost = cost + torch.minimum(1.0 + dist, torch.full_like(dist, float(phrase_len)))
    return cost


def _reconstruct_from_program_bank(
    stage_vector_bank: torch.Tensor,
    program: torch.Tensor,
    group: BlockRVQEncoding,
    flat_indices: torch.Tensor,
    residual_blocks: torch.Tensor,
) -> torch.Tensor:
    block = torch.zeros(int(program.shape[0]), int(stage_vector_bank.shape[2]), dtype=torch.float32)
    for stage_idx in range(int(program.shape[1])):
        block = block + stage_vector_bank[stage_idx][program[:, stage_idx].to(torch.int64)]
    return _apply_block_postprocess(group, block, flat_indices, residual_blocks=residual_blocks)


def _soft_dr_loss(
    model: SemanticISAEncoder,
    features: torch.Tensor,
    sequences: torch.Tensor,
    target_blocks: torch.Tensor,
    init_family_ids: torch.Tensor,
    init_atom_ids: torch.Tensor,
    family_prior: torch.Tensor,
    stage_vector_bank: torch.Tensor,
    group: BlockRVQEncoding,
    flat_indices: torch.Tensor,
    residual_blocks: torch.Tensor,
    args: argparse.Namespace,
    warmup_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    family_logits, atom_logits, _ = model(features)
    family_probs = F.softmax(family_logits / float(args.family_temp), dim=1)
    phrase_probs = F.softmax(model.phrase_logits / float(args.phrase_temp), dim=-1)

    soft_stage_sum = torch.zeros(int(features.shape[0]), group.block_size, dtype=torch.float32)
    program_cost = torch.ones(int(features.shape[0]), dtype=torch.float32)
    correction_load = torch.zeros(int(features.shape[0]), dtype=torch.float32)
    atom_entropy = torch.zeros((), dtype=torch.float32)
    batch_size = int(features.shape[0])
    row_idx = torch.arange(batch_size)
    for slot_idx in range(model.num_slots):
        slot_atom_probs = F.softmax(atom_logits[slot_idx] / float(args.atom_temp), dim=2)
        atom_entropy = atom_entropy + (-(slot_atom_probs.clamp_min(1e-9) * slot_atom_probs.clamp_min(1e-9).log()).sum(dim=2).mean())
        mixture = family_probs.unsqueeze(2) * slot_atom_probs
        slot_phrase_probs = phrase_probs[:, slot_idx]
        token_dist = torch.einsum("nfv,fvpb->npb", mixture, slot_phrase_probs)
        start = slot_idx * model.phrase_len
        end = start + model.phrase_len
        target = sequences[:, start:end].to(torch.int64)
        match_prob = token_dist.gather(2, target.unsqueeze(-1)).squeeze(-1).clamp(0.0, 1.0)
        mismatch = (1.0 - match_prob).sum(dim=1)
        correction_load = correction_load + mismatch
        program_cost = program_cost + torch.minimum(1.0 + mismatch, torch.full_like(mismatch, float(model.phrase_len)))
        for pos in range(model.phrase_len):
            soft_stage_sum = soft_stage_sum + token_dist[:, pos, :] @ stage_vector_bank[start + pos]
    atom_entropy = atom_entropy / float(model.num_slots)

    pred_blocks = _apply_block_postprocess(group, soft_stage_sum, flat_indices, residual_blocks=residual_blocks)
    block_num = (pred_blocks - target_blocks).square().mean(dim=1)
    block_den = target_blocks.square().mean(dim=1).clamp_min(1e-6)
    block_recon_loss = (block_num / block_den).mean()
    program_cost_loss = (program_cost / float(model.num_slots * model.phrase_len)).mean()
    correction_loss = (correction_load / float(model.num_slots * model.phrase_len)).mean()
    low_level_savings = float(model.phrase_len) * float(model.num_slots) + 1.0 - program_cost
    low_level_gain = -(low_level_savings / float(model.num_slots * model.phrase_len)).mean()

    family_entropy = (-(family_probs.clamp_min(1e-9) * family_probs.clamp_min(1e-9).log()).sum(dim=1).mean())
    batch_family_mass = family_probs.mean(dim=0)
    balance_loss = (batch_family_mass - family_prior).square().mean()
    usage_floor_loss = F.relu(float(args.min_family_mass) - batch_family_mass).mean()
    phrase_entropy = (-(phrase_probs.clamp_min(1e-9) * phrase_probs.clamp_min(1e-9).log()).sum(dim=-1).mean())

    norm_features = F.normalize(features, dim=1)
    sim = norm_features @ norm_features.t()
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    contrastive = 1.0 - F.cosine_similarity(family_probs, family_probs[nn_idx], dim=1).mean()

    family_mass = family_probs.sum(dim=0).clamp_min(1e-6)
    centroids = family_probs.t() @ features / family_mass.unsqueeze(1)
    sq_dist = (features[:, None, :] - centroids[None, :, :]).square().mean(dim=2)
    cohesion = (family_probs * sq_dist).sum(dim=1).mean()

    loss = float(args.block_recon_weight) * block_recon_loss
    loss = loss + float(args.program_cost_weight) * program_cost_loss
    loss = loss + float(args.correction_weight) * correction_loss
    loss = loss + float(args.low_level_gain_weight) * low_level_gain
    loss = loss + float(args.contrastive_weight) * contrastive
    loss = loss + float(args.cohesion_weight) * cohesion
    loss = loss + float(args.balance_weight) * balance_loss
    loss = loss + float(args.usage_floor_weight) * usage_floor_loss
    loss = loss + float(args.phrase_entropy_weight) * phrase_entropy
    loss = loss + float(args.assignment_entropy_weight) * (family_entropy + atom_entropy)

    family_ce = torch.zeros((), dtype=torch.float32)
    atom_ce = torch.zeros((), dtype=torch.float32)
    supervision_weight = warmup_weight + float(args.persistent_supervision_weight)
    if supervision_weight > 0.0:
        family_ce = F.cross_entropy(family_logits, init_family_ids)
        for slot_idx in range(model.num_slots):
            selected_atom_logits = atom_logits[slot_idx][row_idx, init_family_ids]
            atom_ce = atom_ce + F.cross_entropy(selected_atom_logits, init_atom_ids[:, slot_idx])
        atom_ce = atom_ce / float(model.num_slots)
        loss = loss + supervision_weight * (family_ce + atom_ce)

    with torch.no_grad():
        phrase_tokens = model.phrase_logits.argmax(dim=-1).to(torch.uint8)
        hard_family = family_logits.argmax(dim=1)
        hard_program = torch.empty_like(sequences)
        for slot_idx in range(model.num_slots):
            selected_atom_logits = atom_logits[slot_idx][row_idx, hard_family]
            hard_atom = selected_atom_logits.argmax(dim=1)
            hard_program[:, slot_idx * model.phrase_len:(slot_idx + 1) * model.phrase_len] = phrase_tokens[:, slot_idx][hard_family, hard_atom]
        hard_token_match = float((hard_program == sequences).to(torch.float32).mean().item())
        hard_cost = _hard_program_cost(hard_program, sequences, model.phrase_len)
        hard_blocks = _reconstruct_from_program_bank(stage_vector_bank, hard_program, group, flat_indices, residual_blocks)
        hard_block_rel_mse = float(
            (((hard_blocks - target_blocks).square().mean(dim=1) / target_blocks.square().mean(dim=1).clamp_min(1e-6)).mean()).item()
        )

    stats = {
        "loss": float(loss.item()),
        "block_recon_loss": float(block_recon_loss.item()),
        "program_cost_loss": float(program_cost_loss.item()),
        "correction_loss": float(correction_loss.item()),
        "low_level_gain": float(low_level_gain.item()),
        "contrastive": float(contrastive.item()),
        "cohesion": float(cohesion.item()),
        "balance": float(balance_loss.item()),
        "usage_floor": float(usage_floor_loss.item()),
        "phrase_entropy": float(phrase_entropy.item()),
        "family_entropy": float(family_entropy.item()),
        "atom_entropy": float(atom_entropy.item()),
        "family_ce": float(family_ce.item()),
        "atom_ce": float(atom_ce.item()),
        "supervision_weight": float(supervision_weight),
        "hard_token_match": hard_token_match,
        "hard_program_cost": float(hard_cost.mean().item()),
        "hard_block_rel_mse": hard_block_rel_mse,
    }
    return loss, stats


def _train_dr_isa(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    if len(enc.groups) != 1:
        raise ValueError("WAL-DR Step 12a currently expects exactly one group per target layer")
    group = enc.groups[0]
    sample_sequences, sample_blocks, sample_scale_features, sample_flat_indices = _sample_training_corpus(
        enc,
        int(args.train_samples),
        seed=0,
    )
    stage_surprisal = (
        -(
            (
                torch.stack(
                    [
                        torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=int(sample_sequences.max().item()) + 1)
                        for pos in range(int(sample_sequences.shape[1]))
                    ],
                    dim=0,
                ).to(torch.float32)
                / max(int(sample_sequences.shape[0]), 1)
            )
            .clamp_min(1e-12)
            .log()
        )
    )
    modal_tokens = torch.stack(
        [
            torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=int(sample_sequences.max().item()) + 1).argmax()
            for pos in range(int(sample_sequences.shape[1]))
        ],
        dim=0,
    ).to(torch.uint8)
    semantic_features = _semantic_features(sample_sequences, stage_surprisal, modal_tokens, int(args.phrase_len))
    raw_features = torch.cat([semantic_features.to(torch.float32), sample_scale_features.to(torch.float32)], dim=1).contiguous()
    family_model = _learn_initial_families(name, sample_sequences, raw_features, args)
    norm_features = family_model["norm_features"].to(torch.float32).contiguous()
    init_family_ids = family_model["sample_family_ids"].to(torch.int64).contiguous()
    base = int(family_model["base"])
    raw_len = int(sample_sequences.shape[1])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len
    init_phrases, init_atom_ids, init_slot_summaries = _build_initial_phrase_bank(sample_sequences, init_family_ids, base, args)
    stage_vector_bank = _prepare_stage_vector_bank(group, base)
    model = SemanticISAEncoder(
        input_dim=int(norm_features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_families=int(args.num_families),
        num_slots=num_slots,
        num_slot_variants=int(args.num_slot_variants),
        phrase_len=phrase_len,
        base=base,
        init_phrases=init_phrases,
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    family_prior = (
        torch.bincount(init_family_ids, minlength=int(args.num_families)).to(torch.float32)
        / max(int(init_family_ids.numel()), 1)
    )
    family_buckets = [(init_family_ids == family_idx).nonzero(as_tuple=True)[0] for family_idx in range(int(args.num_families))]
    history = []
    total_rows = int(sample_sequences.shape[0])
    batch_size = min(int(args.train_batch_size), total_rows)
    log_every = max(1, int(args.log_every))
    t0 = time.time()
    for step in range(1, int(args.train_steps) + 1):
        picks = []
        per_family = max(1, batch_size // int(args.num_families))
        for bucket in family_buckets:
            if int(bucket.numel()) == 0:
                continue
            draw = bucket[torch.randint(0, int(bucket.numel()), (per_family,), generator=generator)]
            picks.append(draw)
        if picks:
            pick = torch.cat(picks, dim=0)
        else:
            pick = torch.empty(0, dtype=torch.int64)
        if int(pick.numel()) < batch_size:
            extra = torch.randint(0, total_rows, (batch_size - int(pick.numel()),), generator=generator)
            pick = torch.cat([pick, extra], dim=0)
        elif int(pick.numel()) > batch_size:
            keep = torch.randperm(int(pick.numel()), generator=generator)[:batch_size]
            pick = pick[keep]
        shuffle = torch.randperm(batch_size, generator=generator)
        pick = pick[shuffle]
        warmup = float(args.init_supervision_weight) * max(0.0, 1.0 - float(step - 1) / max(float(args.warmup_steps), 1.0))
        optimizer.zero_grad(set_to_none=True)
        loss, stats = _soft_dr_loss(
            model,
            norm_features[pick],
            sample_sequences[pick],
            sample_blocks[pick],
            init_family_ids[pick],
            init_atom_ids[pick],
            family_prior,
            stage_vector_bank,
            group,
            sample_flat_indices[pick],
            _selected_residual_blocks(group, sample_flat_indices[pick]),
            args,
            warmup,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == int(args.train_steps):
            dt = time.time() - t0
            entry = {
                "step": int(step),
                "elapsed_s": float(dt),
                "warmup_weight": float(warmup),
                **stats,
            }
            history.append(entry)
            print(
                f"[wal-dr/train] {name}: step={step}/{args.train_steps} loss={stats['loss']:.4f} "
                f"block={stats['block_recon_loss']:.4f} cost={stats['program_cost_loss']:.4f} "
                f"hard_cost={stats['hard_program_cost']:.4f} hard_rel_mse={stats['hard_block_rel_mse']:.4f}",
                flush=True,
            )

    phrase_tokens = model.phrase_logits.detach().argmax(dim=-1).to(torch.uint8).cpu()
    sample_family_orig, _, sample_approx, sample_margin = _predict_hard_from_features(
        model,
        norm_features,
        phrase_tokens,
        int(args.assign_chunk_size),
    )
    num_families = int(args.num_families)
    family_stats = []
    for family_idx in range(num_families):
        mask = sample_family_orig == family_idx
        if bool(mask.any()):
            family_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "modal_match": float(raw_features[mask, 1].mean().item()),
                }
            )
        else:
            family_stats.append(
                {
                    "family_idx": int(family_idx),
                    "avg_surprisal": float("inf"),
                    "modal_match": 0.0,
                }
            )
    order = sorted(family_stats, key=lambda item: (item["avg_surprisal"], -item["modal_match"]))
    family_order = torch.tensor([int(item["family_idx"]) for item in order], dtype=torch.int64)
    family_inverse = torch.empty_like(family_order)
    family_inverse[family_order] = torch.arange(num_families, dtype=torch.int64)
    semantic_family_ids = family_inverse[sample_family_orig]
    family_names = _family_names_for_role(_role_name(name), num_families)
    feature_centroids = torch.zeros(num_families, int(norm_features.shape[1]), dtype=torch.float32)
    sequence_centroids = torch.zeros(num_families, raw_len, dtype=torch.uint8)
    family_sample_summaries = []
    sample_blocks_recon = _reconstruct_from_program_bank(
        stage_vector_bank,
        sample_approx,
        group,
        sample_flat_indices,
        _selected_residual_blocks(group, sample_flat_indices),
    )
    sample_block_rel = ((sample_blocks_recon - sample_blocks).square().mean(dim=1) / sample_blocks.square().mean(dim=1).clamp_min(1e-6)).to(torch.float32)
    for semantic_idx, family_name in enumerate(family_names):
        mask = semantic_family_ids == semantic_idx
        if bool(mask.any()):
            feature_centroids[semantic_idx] = norm_features[mask].mean(dim=0)
            for pos in range(raw_len):
                hist = torch.bincount(sample_sequences[mask, pos].to(torch.int64), minlength=base)
                sequence_centroids[semantic_idx, pos] = int(hist.argmax().item())
            family_sample_summaries.append(
                {
                    "family_id": int(semantic_idx),
                    "family_name": family_name,
                    "sample_count": int(mask.sum().item()),
                    "sample_share": float(mask.to(torch.float32).mean().item()),
                    "sample_avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "sample_modal_match": float(raw_features[mask, 1].mean().item()),
                    "sample_logit_margin": float(sample_margin[mask].mean().item()),
                    "sample_dr_only_token_match": float((sample_approx[mask] == sample_sequences[mask]).to(torch.float32).mean().item()),
                    "sample_dr_only_block_rel_mse": float(sample_block_rel[mask].mean().item()),
                }
            )
        else:
            feature_centroids[semantic_idx] = family_model["feature_centroids"][semantic_idx]
            sequence_centroids[semantic_idx] = family_model["sequence_centroids"][semantic_idx]
            family_sample_summaries.append(
                {
                    "family_id": int(semantic_idx),
                    "family_name": family_name,
                    "sample_count": 0,
                    "sample_share": 0.0,
                    "sample_avg_surprisal": 0.0,
                    "sample_modal_match": 0.0,
                    "sample_logit_margin": 0.0,
                    "sample_dr_only_token_match": 0.0,
                    "sample_dr_only_block_rel_mse": 0.0,
                }
            )

    return {
        "model": model,
        "group": group,
        "base": base,
        "raw_len": raw_len,
        "num_slots": num_slots,
        "sample_sequences": sample_sequences,
        "stage_surprisal": stage_surprisal,
        "modal_tokens": modal_tokens,
        "feature_mean": family_model["feature_mean"].to(torch.float32),
        "feature_std": family_model["feature_std"].to(torch.float32),
        "feature_centroids": feature_centroids,
        "sequence_centroids": sequence_centroids,
        "phrase_tokens": phrase_tokens,
        "family_order": family_order,
        "family_inverse": family_inverse,
        "family_names": family_names,
        "training_history": history,
        "family_sample_summaries": family_sample_summaries,
        "init_slot_summaries": init_slot_summaries,
        "sample_token_match": float((sample_approx == sample_sequences).to(torch.float32).mean().item()),
        "sample_block_rel_mse": float(sample_block_rel.mean().item()),
        "sample_family_entropy": float(
            -(
                (torch.bincount(semantic_family_ids, minlength=num_families).to(torch.float64) / max(int(sample_sequences.shape[0]), 1)).clamp_min(1e-12)
                * (torch.bincount(semantic_family_ids, minlength=num_families).to(torch.float64) / max(int(sample_sequences.shape[0]), 1)).clamp_min(1e-12).log()
            ).sum().item()
        ),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def _build_dr_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    trained = _train_dr_isa(name, enc, args)
    group = trained["group"]
    model = trained["model"]
    phrase_tokens = trained["phrase_tokens"]
    family_inverse = trained["family_inverse"]
    family_names = trained["family_names"]
    feature_mean = trained["feature_mean"]
    feature_std = trained["feature_std"]
    feature_centroids = trained["feature_centroids"]
    sequence_centroids = trained["sequence_centroids"]
    stage_surprisal = trained["stage_surprisal"]
    modal_tokens = trained["modal_tokens"]
    raw_len = int(trained["raw_len"])
    phrase_len = int(args.phrase_len)
    num_slots = raw_len // phrase_len
    num_families = int(args.num_families)

    seq_mat = _flatten_group_sequences(group)
    rows = int(seq_mat.shape[0])
    flat_all = torch.arange(rows, dtype=torch.int64)
    scale_features = _selected_scale_features(group, flat_all)
    approx_program = torch.empty_like(seq_mat)
    group_low_level_calls = torch.zeros(rows, dtype=torch.int32)
    group_literal_corrections = torch.zeros(rows, dtype=torch.int32)
    group_program_units = torch.ones(rows, dtype=torch.int32)
    group_family_ids = torch.empty(rows, dtype=torch.int64)
    num_tokens = rows * raw_len
    family_count_totals = torch.zeros(num_families, dtype=torch.int64)
    family_surprisal_sum = torch.zeros(num_families, dtype=torch.float64)
    family_modal_match_sum = torch.zeros(num_families, dtype=torch.float64)
    family_feature_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_sequence_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_margin_sum = torch.zeros(num_families, dtype=torch.float64)
    family_logit_margin_sum = torch.zeros(num_families, dtype=torch.float64)
    family_low_level_calls = torch.zeros(num_families, dtype=torch.int64)
    family_literal_corrections = torch.zeros(num_families, dtype=torch.int64)
    family_program_units = torch.zeros(num_families, dtype=torch.int64)
    family_low_level_tokens = torch.zeros(num_families, dtype=torch.int64)
    family_token_match_hamming = torch.zeros(num_families, dtype=torch.int64)
    family_instruction_usage: Counter[tuple[int, int, int]] = Counter()
    total_e2e_hamming = 0
    semantic_weight = float(args.report_semantic_weight)
    sequence_weight = float(args.report_sequence_weight)
    for start in range(0, rows, int(args.assign_chunk_size)):
        end = min(start + int(args.assign_chunk_size), rows)
        chunk = seq_mat[start:end]
        raw_features = torch.cat(
            [
                _semantic_features(chunk, stage_surprisal, modal_tokens, phrase_len).to(torch.float32),
                scale_features[start:end].to(torch.float32),
            ],
            dim=1,
        )
        norm_features = ((raw_features - feature_mean) / feature_std).to(torch.float32)
        hard_family_orig, hard_atom_ids, approx_chunk, logit_margin = _predict_hard_from_features(
            model,
            norm_features,
            phrase_tokens,
            int(args.assign_chunk_size),
        )
        semantic_family_ids = family_inverse[hard_family_orig]
        group_family_ids[start:end] = semantic_family_ids
        approx_program[start:end] = approx_chunk
        feature_dist = (norm_features[:, None, :] - feature_centroids[None, :, :]).square().mean(dim=2)
        sequence_dist = (chunk[:, None, :] != sequence_centroids[None, :, :]).to(torch.float32).mean(dim=2)
        combined = semantic_weight * feature_dist + sequence_weight * sequence_dist
        own_idx = semantic_family_ids.unsqueeze(1)
        own_feature = feature_dist.gather(1, own_idx).squeeze(1)
        own_sequence = sequence_dist.gather(1, own_idx).squeeze(1)
        own_combined = combined.gather(1, own_idx).squeeze(1)
        if num_families > 1:
            other = combined.clone()
            other.scatter_(1, own_idx, float("inf"))
            margin = other.min(dim=1).values - own_combined
        else:
            margin = torch.zeros_like(own_combined)
        for slot_idx in range(num_slots):
            start_pos = slot_idx * phrase_len
            end_pos = start_pos + phrase_len
            dist_i32 = (chunk[:, start_pos:end_pos] != approx_chunk[:, start_pos:end_pos]).sum(dim=1).to(torch.int32)
            use_atom = (dist_i32 + 1) < phrase_len
            saved_tokens = torch.where(use_atom, phrase_len - dist_i32, torch.zeros_like(dist_i32))
            group_low_level_calls[start:end] += use_atom.to(torch.int32)
            group_literal_corrections[start:end] += torch.where(use_atom, dist_i32, torch.full_like(dist_i32, phrase_len))
            group_program_units[start:end] += torch.where(use_atom, dist_i32 + 1, torch.full_like(dist_i32, phrase_len))
            total_e2e_hamming += int(dist_i32.sum().item())
            for family_idx in range(num_families):
                mask = semantic_family_ids == family_idx
                if not bool(mask.any()):
                    continue
                family_low_level_tokens[family_idx] += int(saved_tokens[mask].sum().item())
                family_token_match_hamming[family_idx] += int(dist_i32[mask].sum().item())
                selected_atom = hard_atom_ids[mask, slot_idx]
                selected_use_atom = use_atom[mask]
                if bool(selected_use_atom.any()):
                    slot_counts = torch.bincount(
                        selected_atom[selected_use_atom],
                        minlength=int(args.num_slot_variants),
                    )
                    for variant_idx, count in enumerate(slot_counts.tolist()):
                        if count > 0:
                            family_instruction_usage[(int(family_idx), int(slot_idx), int(variant_idx))] += int(count)
        for family_idx in range(num_families):
            mask = semantic_family_ids == family_idx
            if not bool(mask.any()):
                continue
            count = int(mask.sum().item())
            family_count_totals[family_idx] += count
            family_surprisal_sum[family_idx] += float(raw_features[mask, 0].sum().item())
            family_modal_match_sum[family_idx] += float(raw_features[mask, 1].sum().item())
            family_feature_radius_sum[family_idx] += float(own_feature[mask].sum().item())
            family_sequence_radius_sum[family_idx] += float(own_sequence[mask].sum().item())
            family_margin_sum[family_idx] += float(margin[mask].sum().item())
            family_logit_margin_sum[family_idx] += float(logit_margin[mask].sum().item())

    total_low_level_calls = int(group_low_level_calls.sum().item())
    total_literal_corrections = int(group_literal_corrections.sum().item())
    total_program_units = int(group_program_units.sum().item())
    for family_idx in range(num_families):
        mask = group_family_ids == family_idx
        if not bool(mask.any()):
            continue
        family_low_level_calls[family_idx] += int(group_low_level_calls[mask].sum().item())
        family_literal_corrections[family_idx] += int(group_literal_corrections[mask].sum().item())
        family_program_units[family_idx] += int(group_program_units[mask].sum().item())

    family_probs = family_count_totals.to(torch.float64) / max(rows, 1)
    family_entropy = float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item())
    approx_group = _clone_group_with_program_matrix(group, approx_program)
    dr_only = GroupedBlockRVQEncoding(groups=(approx_group,), row_slices=enc.row_slices, original_shape=enc.original_shape)
    family_summaries = []
    for family_idx, family_name in enumerate(family_names):
        count = int(family_count_totals[family_idx].item())
        atom_usage_rows = []
        for slot_idx in range(num_slots):
            for variant_idx in range(int(args.num_slot_variants)):
                atom_usage = int(family_instruction_usage.get((family_idx, slot_idx, variant_idx), 0))
                if atom_usage > 0:
                    atom_usage_rows.append(
                        {
                            "instruction_name": _instruction_name(family_name, slot_idx, variant_idx),
                            "slot_index": int(slot_idx),
                            "variant_id": int(variant_idx),
                            "used_calls": atom_usage,
                        }
                    )
        atom_usage_rows.sort(key=lambda item: item["used_calls"], reverse=True)
        if count > 0:
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "block_count": count,
                    "block_share": float(count / max(rows, 1)),
                    "avg_surprisal": float(family_surprisal_sum[family_idx].item() / count),
                    "avg_modal_match": float(family_modal_match_sum[family_idx].item() / count),
                    "avg_feature_radius": float(family_feature_radius_sum[family_idx].item() / count),
                    "avg_sequence_radius": float(family_sequence_radius_sum[family_idx].item() / count),
                    "avg_margin": float(family_margin_sum[family_idx].item() / count),
                    "avg_logit_margin": float(family_logit_margin_sum[family_idx].item() / count),
                    "avg_low_level_calls": float(family_low_level_calls[family_idx].item() / count),
                    "avg_literal_corrections": float(family_literal_corrections[family_idx].item() / count),
                    "avg_program_length": float(family_program_units[family_idx].item() / count),
                    "low_level_token_coverage": float(family_low_level_tokens[family_idx].item() / max(count * raw_len, 1)),
                    "dr_only_token_match": float(1.0 - family_token_match_hamming[family_idx].item() / max(count * raw_len, 1)),
                    "top_atoms": atom_usage_rows[:10],
                }
            )
        else:
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "block_count": 0,
                    "block_share": 0.0,
                    "avg_surprisal": 0.0,
                    "avg_modal_match": 0.0,
                    "avg_feature_radius": 0.0,
                    "avg_sequence_radius": 0.0,
                    "avg_margin": 0.0,
                    "avg_logit_margin": 0.0,
                    "avg_low_level_calls": 0.0,
                    "avg_literal_corrections": 0.0,
                    "avg_program_length": 0.0,
                    "low_level_token_coverage": 0.0,
                    "dr_only_token_match": 0.0,
                    "top_atoms": [],
                }
            )

    top_atoms = []
    for (family_idx, slot_idx, variant_idx), used_calls in family_instruction_usage.most_common(12):
        top_atoms.append(
            {
                "instruction_name": _instruction_name(family_names[family_idx], slot_idx, variant_idx),
                "family_name": family_names[family_idx],
                "slot_index": int(slot_idx),
                "variant_id": int(variant_idx),
                "used_calls": int(used_calls),
            }
        )

    exact_stats = {
        "name": name,
        "raw_program_length": int(raw_len),
        "avg_program_length": float(total_program_units / max(rows, 1)),
        "avg_high_level_calls": 1.0,
        "avg_low_level_calls": float(total_low_level_calls / max(rows, 1)),
        "avg_literal_corrections": float(total_literal_corrections / max(rows, 1)),
        "low_level_token_coverage": float(family_low_level_tokens.sum().item() / max(num_tokens, 1)),
        "hierarchical_instruction_share": float((rows + total_low_level_calls) / max(total_program_units, 1)),
        "program_compression_ratio": float(total_program_units / max(num_tokens, 1)),
        "instruction_vocab_size": int(num_families + num_families * num_slots * int(args.num_slot_variants)),
        "high_level_instruction_count": int(num_families),
        "low_level_instruction_count": int(num_families * num_slots * int(args.num_slot_variants)),
        "active_family_count": int(sum(1 for count in family_count_totals.tolist() if count > 0)),
        "family_entropy": float(family_entropy),
        "dr_only_avg_hamming": float(total_e2e_hamming / max(rows, 1)),
        "dr_only_token_match": float(1.0 - total_e2e_hamming / max(num_tokens, 1)),
        "sample_dr_only_token_match": float(trained["sample_token_match"]),
        "sample_dr_only_block_rel_mse": float(trained["sample_block_rel_mse"]),
        "sample_family_entropy": float(trained["sample_family_entropy"]),
        "training_history": trained["training_history"],
        "family_sample_summaries": trained["family_sample_summaries"],
        "family_summaries": family_summaries,
        "init_slot_summaries": trained["init_slot_summaries"],
        "top_atoms": top_atoms,
    }
    artifact = {
        "family_names": family_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_centroids": feature_centroids,
        "sequence_centroids": sequence_centroids,
        "phrase_tokens": phrase_tokens[trained["family_order"]].clone(),
        "family_order": trained["family_order"],
        "family_inverse": trained["family_inverse"],
        "family_sample_summaries": trained["family_sample_summaries"],
        "training_history": trained["training_history"],
        "state_dict": trained["state_dict"],
    }
    return dr_only, exact_stats, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", choices=("raw", "local"), default="local")
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--num-families", type=int, default=4)
    parser.add_argument("--num-slot-variants", type=int, default=4)
    parser.add_argument("--phrase-len", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=65536)
    parser.add_argument("--atom-iters", type=int, default=6)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--init-family-iters", type=int, default=4)
    parser.add_argument("--init-semantic-weight", type=float, default=1.0)
    parser.add_argument("--init-sequence-weight", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--init-supervision-weight", type=float, default=0.20)
    parser.add_argument("--family-temp", type=float, default=1.0)
    parser.add_argument("--atom-temp", type=float, default=1.0)
    parser.add_argument("--phrase-temp", type=float, default=1.0)
    parser.add_argument("--block-recon-weight", type=float, default=1.0)
    parser.add_argument("--program-cost-weight", type=float, default=0.20)
    parser.add_argument("--correction-weight", type=float, default=0.15)
    parser.add_argument("--low-level-gain-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.08)
    parser.add_argument("--cohesion-weight", type=float, default=0.06)
    parser.add_argument("--balance-weight", type=float, default=0.10)
    parser.add_argument("--usage-floor-weight", type=float, default=0.20)
    parser.add_argument("--min-family-mass", type=float, default=0.06)
    parser.add_argument("--phrase-entropy-weight", type=float, default=0.005)
    parser.add_argument("--assignment-entropy-weight", type=float, default=0.01)
    parser.add_argument("--persistent-supervision-weight", type=float, default=0.03)
    parser.add_argument("--report-semantic-weight", type=float, default=1.0)
    parser.add_argument("--report-sequence-weight", type=float, default=0.35)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_e2e_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_dr_current_l54_gate_up.pt"))
    parser.add_argument("--dr-only-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_dr_only_l54_gate_up.pt"))
    parser.add_argument("--dr-artifact", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_dr_isa_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_dr_proto_summary.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)

    dr_only_encodings: dict[str, GroupedBlockRVQEncoding] = {}
    wal_dr_rows = []
    dr_only_compare = []
    artifacts = {}
    for name in TARGETS:
        dr_only, exact_stats, artifact = _build_dr_layer(name, current_enc[name], args)
        dr_only_encodings[name] = dr_only
        wal_dr_rows.append(exact_stats)
        dr_only_compare.append({"name": name, **_compare_encodings(current_enc[name], dr_only)})
        artifacts[name] = artifact
        print(
            f"[wal-dr] {name}: avg_program={exact_stats['avg_program_length']:.3f}/{exact_stats['raw_program_length']} "
            f"low_calls={exact_stats['avg_low_level_calls']:.3f} entropy={exact_stats['family_entropy']:.3f} "
            f"dr_only_match={exact_stats['dr_only_token_match']:.3f} sample_block_rel={exact_stats['sample_dr_only_block_rel_mse']:.3f}",
            flush=True,
        )

    dr_artifact = Path(args.dr_artifact)
    dr_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(TARGETS),
            "num_families": int(args.num_families),
            "num_slot_variants": int(args.num_slot_variants),
            "train_steps": int(args.train_steps),
            "artifacts": artifacts,
        },
        dr_artifact,
    )
    dr_only_cache = Path(args.dr_only_cache)
    save_grouped_encoding_map(dr_only_cache, dr_only_encodings)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] dr_only via {args.matmul_strategy}", flush=True)
    dr_only_eval = _run_preencoded_eval(ids, dr_only_cache, args.matmul_strategy, args.num_windows)
    exact_eval = dict(current_eval)
    exact_eval["by_construction_identical"] = True

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "num_families": int(args.num_families),
        "num_slot_variants": int(args.num_slot_variants),
        "phrase_len": int(args.phrase_len),
        "train_steps": int(args.train_steps),
        "hidden_dim": int(args.hidden_dim),
        "dense": dense,
        "current_cache": str(current_cache),
        "dr_only_cache": str(dr_only_cache),
        "dr_artifact": str(dr_artifact),
        "wal_dr": wal_dr_rows,
        "dr_only_compare": dr_only_compare,
        "current_eval": current_eval,
        "exact_eval": exact_eval,
        "dr_only_eval": dr_only_eval,
        "delta_exact": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
        },
        "delta_dr_only": {
            "ppl_delta": float(dr_only_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(dr_only_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(dr_only_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  dense:        ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:      ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_dr_exact: ppl={exact_eval['metrics']['perplexity']:.4f} tok/s={exact_eval['metrics']['tok_s']:.2f} peak_mb={exact_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  dr_only:      ppl={dr_only_eval['metrics']['perplexity']:.4f} tok/s={dr_only_eval['metrics']['tok_s']:.2f} peak_mb={dr_only_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row, compare in zip(wal_dr_rows, dr_only_compare):
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"low_calls={row['avg_low_level_calls']:.3f} family_entropy={row['family_entropy']:.3f} "
            f"dr_only_rel_mse={compare['recon_rel_mse']:.6f}",
            flush=True,
        )
    print(
        f"  delta(dr_only-current): ppl={result['delta_dr_only']['ppl_delta']:+.6f} "
        f"tok/s={result['delta_dr_only']['tok_s_delta']:+.2f} peak_mb={result['delta_dr_only']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()