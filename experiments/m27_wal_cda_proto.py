"""M27 WAL-CDA Step 12f: constrained context-dependent atom deltas.

This probe keeps the old WAL-LHA / WAL-SBC semantic assignments fixed and only
adds a tiny context-aware low-rank delta on top of the old atom basis.

Key guardrails:
  - no raw numeric IDs as coordinates; all discrete signals use embeddings
  - context enters only through a cheap detached activation summary
  - delta is factorized low-rank and clipped, not a full decoder head
  - usefulness is optimized through a soft expected-cost surrogate under the
    same budget contract, against the best old accepted path on the old basis
  - evaluation preserves three explicit modes:
      1. strict_legacy
      2. budgeted_exact_old
      3. budgeted_exact_cda
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from m27_wal_dr_proto import _apply_block_postprocess, _sample_training_corpus
from m27_wal_lha_proto import (
    MODEL_DIR,
    ROOT,
    TARGETS,
    _activation_chunks,
    _build_activation_bank,
    _build_current_cache,
    _clone_group_with_program_matrix,
    _compare_encodings,
    _eval_ids,
    _family_names_for_role,
    _flatten_group_sequences,
    _output_rel_mse_vector,
    _predict_lha_hard_from_features,
    _prepare_stage_vector_bank,
    _reconstruct_from_program_bank,
    _role_name,
    _run_dense_eval,
    _run_preencoded_eval,
    _selected_residual_blocks,
    _selected_scale_features,
    _semantic_features,
    _train_lha_isa,
)
from m27_wal_sbc_core import (
    best_residual_match,
    build_residual_phrase_bank,
    prepare_slot_phrase_block_bank,
    slot_block_rel_mse,
    slot_blocks_from_tokens,
    slot_output_rel_mse,
)
from m27_wal_sbc_offline import load_sbc_artifacts
from m27_wal_sbc_proto import DEFAULTS as SBC_DEFAULTS, _collect_layer_inputs_budgeted
from dwl2_dynamic_route.src.block_vq import GroupedBlockRVQEncoding
from dwl2_dynamic_route.src.encoding_io import save_grouped_encoding_map


DEFAULTS = {
    **SBC_DEFAULTS,
    "current_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_current_l54_gate_up.pt"),
    "old_budgeted_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_cda_old_budgeted_l54_gate_up.pt"),
    "cda_budgeted_cache": str(ROOT / "dwl2_dynamic_route/results/m27_wal_cda_budgeted_l54_gate_up.pt"),
    "sbc_artifact": str(ROOT / "dwl2_dynamic_route/results/m27_wal_sbc_isa_l54_gate_up.pt"),
    "cda_artifact": str(ROOT / "dwl2_dynamic_route/results/m27_wal_cda_isa_l54_gate_up.pt"),
    "out": str(ROOT / "dwl2_dynamic_route/results/m27_wal_cda_proto_summary.json"),
    "ctx_dim": 16,
    "cda_hidden_dim": 64,
    "rank": 8,
    "summary_dim": 8,
    "delta_clip": 0.75,
    "phrase_temp": 1.0,
    "usefulness_tau": 0.10,
    "atom_cost": 1.0,
    "literal_cost": 3.0,
    "output_recon_weight": 1.0,
    "block_aux_weight": 0.20,
    "usefulness_weight": 0.45,
    "delta_norm_weight": 0.05,
    "train_steps": 220,
    "train_batch_size": 1024,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "log_every": 20,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", choices=("raw", "local"))
    parser.add_argument("--num-windows", type=int)
    parser.add_argument("--lha-calibration-windows", type=int)
    parser.add_argument("--capture-max-len", type=int)
    parser.add_argument("--activation-rows", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--assign-chunk-size", type=int)
    parser.add_argument("--matmul-strategy")
    parser.add_argument("--atom-block-budget", type=float)
    parser.add_argument("--atom-output-budget", type=float)
    parser.add_argument("--residual-block-budget", type=float)
    parser.add_argument("--residual-output-budget", type=float)
    parser.add_argument("--residual-bank-size", type=int)
    parser.add_argument("--residual-program-cost", type=float)
    parser.add_argument("--ctx-dim", type=int)
    parser.add_argument("--cda-hidden-dim", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--summary-dim", type=int)
    parser.add_argument("--delta-clip", type=float)
    parser.add_argument("--phrase-temp", type=float)
    parser.add_argument("--usefulness-tau", type=float)
    parser.add_argument("--atom-cost", type=float)
    parser.add_argument("--literal-cost", type=float)
    parser.add_argument("--output-recon-weight", type=float)
    parser.add_argument("--block-aux-weight", type=float)
    parser.add_argument("--usefulness-weight", type=float)
    parser.add_argument("--delta-norm-weight", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--current-cache")
    parser.add_argument("--old-budgeted-cache")
    parser.add_argument("--cda-budgeted-cache")
    parser.add_argument("--sbc-artifact")
    parser.add_argument("--cda-artifact")
    parser.add_argument("--out")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    raw = parser.parse_args()
    config = dict(DEFAULTS)
    for key, value in vars(raw).items():
        if value is not None and value is not False:
            config[key] = value
    if raw.rebuild_cache:
        config["rebuild_cache"] = True
    if raw.calibrate_stage_scales:
        config["calibrate_stage_scales"] = True
    return argparse.Namespace(**config)


def _stage_feature_stats(enc: GroupedBlockRVQEncoding, train_samples: int, phrase_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample_sequences, _, _, _ = _sample_training_corpus(enc, train_samples, seed=0)
    base = int(sample_sequences.max().item()) + 1
    stage_surprisal = -(
        (
            torch.stack(
                [torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base) for pos in range(int(sample_sequences.shape[1]))],
                dim=0,
            ).to(torch.float32)
            / max(int(sample_sequences.shape[0]), 1)
        )
        .clamp_min(1e-12)
        .log()
    )
    modal_tokens = torch.stack(
        [torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base).argmax() for pos in range(int(sample_sequences.shape[1]))],
        dim=0,
    ).to(torch.uint8)
    return stage_surprisal, modal_tokens


def _activation_summary_from_chunks(act_chunks: torch.Tensor) -> torch.Tensor:
    act = act_chunks.detach().to(torch.float32)
    row_norms = act.square().mean(dim=2).sqrt()
    topk = min(4, int(row_norms.shape[1]))
    topk_vals = row_norms.topk(k=topk, dim=1).values
    summary = torch.stack(
        [
            act.mean(dim=(1, 2)),
            act.square().mean(dim=(1, 2)).sqrt(),
            act.abs().mean(dim=(1, 2)),
            act.abs().amax(dim=(1, 2)),
            row_norms.mean(dim=1),
            row_norms.std(dim=1, correction=0),
            topk_vals.mean(dim=1),
            topk_vals[:, 0],
        ],
        dim=1,
    )
    return summary.contiguous()


class CDAAtomGenerator(nn.Module):
    def __init__(
        self,
        base_phrase_logits: torch.Tensor,
        num_slots: int,
        ctx_summary_dim: int,
        ctx_dim: int,
        hidden_dim: int,
        rank: int,
        delta_clip: float,
    ) -> None:
        super().__init__()
        base_logits = base_phrase_logits.detach().to(torch.float32)
        self.register_buffer("base_phrase_logits", base_logits)
        self.register_buffer("base_stage_ids", base_logits.argmax(dim=-1).to(torch.int64))
        self.num_families = int(base_logits.shape[0])
        self.num_slots = int(base_logits.shape[1])
        self.num_variants = int(base_logits.shape[2])
        self.phrase_len = int(base_logits.shape[3])
        self.base = int(base_logits.shape[4])
        self.delta_clip = float(delta_clip)

        self.stage_emb = nn.Embedding(self.base, ctx_dim)
        self.family_emb = nn.Embedding(self.num_families, ctx_dim)
        self.variant_emb = nn.Embedding(self.num_variants, ctx_dim)
        self.slot_emb = nn.Embedding(num_slots, ctx_dim)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim * (self.phrase_len + 3) + ctx_summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, rank)
        self.delta_basis = nn.Parameter(torch.randn(rank, self.phrase_len, self.base) * 0.02)
        self.conf_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        family_ids: torch.Tensor,
        variant_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        layer_input_summary: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_logits = self.base_phrase_logits[family_ids, slot_ids, variant_ids]
        base_stage_ids = self.base_stage_ids[family_ids, slot_ids, variant_ids]
        stage_ctx = self.stage_emb(base_stage_ids).reshape(int(base_stage_ids.shape[0]), -1)
        ctx = torch.cat(
            [
                stage_ctx,
                self.family_emb(family_ids),
                self.variant_emb(variant_ids),
                self.slot_emb(slot_ids),
                layer_input_summary,
            ],
            dim=1,
        )
        hidden = self.ctx_mlp(ctx)
        alpha = self.alpha_head(hidden)
        delta_raw = torch.einsum("nr,rpb->npb", alpha, self.delta_basis)
        delta = self.delta_clip * torch.tanh(delta_raw)
        confidence = self.conf_head(hidden).squeeze(1)
        return {
            "logits": base_logits + delta,
            "base_logits": base_logits,
            "delta": delta,
            "confidence": confidence,
            "hidden": hidden,
            "alpha": alpha,
        }


def _load_or_build_old_basis(
    artifact_path: Path,
    current_enc: dict[str, GroupedBlockRVQEncoding],
    activation_banks: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    if artifact_path.exists():
        return load_sbc_artifacts(artifact_path)

    states: dict[str, dict[str, object]] = {}
    for name in TARGETS:
        trained = _train_lha_isa(name, current_enc[name], activation_banks[name], args)
        residual_bank, residual_bank_summaries = build_residual_phrase_bank(
            trained["sample_sequences"],
            int(args.phrase_len),
            int(trained["num_slots"]),
            int(args.residual_bank_size),
        )
        states[name] = {
            **trained,
            "residual_phrase_bank": residual_bank,
            "residual_bank_summaries": residual_bank_summaries,
        }
    return states


def _prepare_cda_surface(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    old_state: dict[str, object],
    args: argparse.Namespace,
    sample_only: bool,
) -> dict[str, object]:
    group = enc.groups[0]
    phrase_len = int(old_state["base_phrase_tokens"].shape[3])
    num_slots = int(old_state["base_phrase_tokens"].shape[1])
    if sample_only:
        sequences, target_blocks, scale_features, flat_indices = _sample_training_corpus(enc, int(args.train_samples), seed=0)
    else:
        sequences = _flatten_group_sequences(group)
        flat_indices = torch.arange(int(sequences.shape[0]), dtype=torch.int64)
        scale_features = _selected_scale_features(group, flat_indices)
        residual_blocks = _selected_residual_blocks(group, flat_indices)
        stage_vector_bank = _prepare_stage_vector_bank(group, int(old_state["model"].base))
        target_blocks = _reconstruct_from_program_bank(stage_vector_bank, sequences, group, flat_indices, residual_blocks)

    stage_surprisal, modal_tokens = _stage_feature_stats(enc, int(args.train_samples), phrase_len)
    stage_vector_bank = _prepare_stage_vector_bank(group, int(old_state["model"].base))
    residual_bank = old_state["residual_phrase_bank"].to(torch.uint8)
    residual_block_bank = prepare_slot_phrase_block_bank(stage_vector_bank, residual_bank, phrase_len)
    rows = int(sequences.shape[0])
    raw_len = int(sequences.shape[1])
    blocks_per_row = int(group.stage_shape[1])

    old_family_orig = torch.empty(rows, dtype=torch.int64)
    semantic_family_ids = torch.empty(rows, dtype=torch.int64)
    old_atom_ids = torch.empty(rows, num_slots, dtype=torch.int64)
    old_approx_program = torch.empty_like(sequences)
    residual_program = torch.empty_like(sequences)
    activation_summary = torch.empty(rows, int(args.summary_dim), dtype=torch.float32)
    old_atom_block_rel = torch.empty(rows, num_slots, dtype=torch.float32)
    old_atom_output_rel = torch.empty(rows, num_slots, dtype=torch.float32)
    old_residual_block_rel = torch.empty(rows, num_slots, dtype=torch.float32)
    old_residual_output_rel = torch.empty(rows, num_slots, dtype=torch.float32)
    old_cost = torch.empty(rows, num_slots, dtype=torch.float32)
    family_counts = torch.zeros(len(old_state["family_names"]), dtype=torch.int64)
    atom_cost = float(args.atom_cost)
    residual_cost = atom_cost + float(args.residual_program_cost)
    literal_cost = float(args.literal_cost)

    for start in range(0, rows, int(args.assign_chunk_size)):
        end = min(start + int(args.assign_chunk_size), rows)
        chunk = sequences[start:end]
        flat_chunk = flat_indices[start:end]
        raw_features = torch.cat(
            [
                _semantic_features(chunk, stage_surprisal, modal_tokens, phrase_len).to(torch.float32),
                scale_features[start:end].to(torch.float32),
            ],
            dim=1,
        )
        norm_features = ((raw_features - old_state["feature_mean"]) / old_state["feature_std"]).to(torch.float32)
        hard_family_orig, hard_atom, _, approx_chunk, _, _ = _predict_lha_hard_from_features(
            old_state["model"],
            norm_features,
            int(args.assign_chunk_size),
        )
        old_family_orig[start:end] = hard_family_orig
        old_atom_ids[start:end] = hard_atom
        semantic = old_state["family_inverse"][hard_family_orig]
        semantic_family_ids[start:end] = semantic
        family_counts += torch.bincount(semantic, minlength=len(old_state["family_names"]))
        old_approx_program[start:end] = approx_chunk
        act_chunks = _activation_chunks(activation_bank, flat_chunk, blocks_per_row)
        activation_summary[start:end] = _activation_summary_from_chunks(act_chunks)
        for slot_idx in range(num_slots):
            pos0 = slot_idx * phrase_len
            pos1 = pos0 + phrase_len
            target_tokens = chunk[:, pos0:pos1]
            atom_tokens = approx_chunk[:, pos0:pos1]
            target_slot_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, target_tokens)
            atom_slot_blocks = slot_blocks_from_tokens(stage_vector_bank, slot_idx, phrase_len, atom_tokens)
            _, residual_tokens, residual_block_rel, residual_output_rel = best_residual_match(
                act_chunks,
                target_slot_blocks,
                residual_bank[slot_idx],
                residual_block_bank[slot_idx],
            )
            atom_block_rel = slot_block_rel_mse(atom_slot_blocks, target_slot_blocks)
            atom_output_rel = slot_output_rel_mse(act_chunks, atom_slot_blocks, target_slot_blocks)
            residual_program[start:end, pos0:pos1] = residual_tokens
            old_atom_block_rel[start:end, slot_idx] = atom_block_rel
            old_atom_output_rel[start:end, slot_idx] = atom_output_rel
            old_residual_block_rel[start:end, slot_idx] = residual_block_rel
            old_residual_output_rel[start:end, slot_idx] = residual_output_rel

            atom_accept = atom_output_rel <= float(args.atom_output_budget)
            if args.atom_block_budget is not None:
                atom_accept &= atom_block_rel <= float(args.atom_block_budget)
            residual_accept = (~atom_accept) & (residual_output_rel <= float(args.residual_output_budget))
            if args.residual_block_budget is not None:
                residual_accept &= residual_block_rel <= float(args.residual_block_budget)
            slot_cost = torch.full_like(atom_output_rel, literal_cost)
            slot_cost[atom_accept] = atom_cost
            slot_cost[residual_accept] = residual_cost
            old_cost[start:end, slot_idx] = slot_cost

    family_probs = family_counts.to(torch.float64) / max(rows, 1)
    family_entropy = float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item())
    return {
        "name": name,
        "group": group,
        "rows": rows,
        "raw_len": raw_len,
        "phrase_len": phrase_len,
        "num_slots": num_slots,
        "sequences": sequences,
        "target_blocks": target_blocks,
        "flat_indices": flat_indices,
        "blocks_per_row": blocks_per_row,
        "stage_vector_bank": stage_vector_bank,
        "old_family_orig": old_family_orig,
        "semantic_family_ids": semantic_family_ids,
        "old_atom_ids": old_atom_ids,
        "old_approx_program": old_approx_program,
        "residual_program": residual_program,
        "activation_summary": activation_summary,
        "old_atom_block_rel": old_atom_block_rel,
        "old_atom_output_rel": old_atom_output_rel,
        "old_residual_block_rel": old_residual_block_rel,
        "old_residual_output_rel": old_residual_output_rel,
        "old_cost": old_cost,
        "family_entropy": family_entropy,
    }


def _soft_usefulness_loss(
    atom_output_err: torch.Tensor,
    residual_output_err: torch.Tensor,
    old_cost: torch.Tensor,
    atom_budget: float,
    residual_budget: float,
    atom_cost: float,
    residual_cost: float,
    literal_cost: float,
    conf_logit: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    p_atom = torch.sigmoid((float(atom_budget) - atom_output_err) / max(float(tau), 1e-6) + conf_logit)
    p_res = torch.sigmoid((float(residual_budget) - residual_output_err) / max(float(tau), 1e-6))
    expected_cda_cost = p_atom * float(atom_cost) + (1.0 - p_atom) * (
        p_res * float(residual_cost) + (1.0 - p_res) * float(literal_cost)
    )
    gain = old_cost - expected_cda_cost
    return -gain.clamp_min(0.0).mean(), {
        "p_atom_mean": float(p_atom.mean().item()),
        "p_res_mean": float(p_res.mean().item()),
        "gain_mean": float(gain.mean().item()),
        "expected_cda_cost": float(expected_cda_cost.mean().item()),
    }


def _train_cda_adapter(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    old_state: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    surface = _prepare_cda_surface(name, enc, activation_bank, old_state, args, sample_only=True)
    model = CDAAtomGenerator(
        base_phrase_logits=old_state["model"].atom_generator.base_phrase_logits,
        num_slots=int(surface["num_slots"]),
        ctx_summary_dim=int(args.summary_dim),
        ctx_dim=int(args.ctx_dim),
        hidden_dim=int(args.cda_hidden_dim),
        rank=int(args.rank),
        delta_clip=float(args.delta_clip),
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    total_rows = int(surface["rows"])
    batch_size = min(int(args.train_batch_size), total_rows)
    log_every = max(1, int(args.log_every))
    history = []
    t0 = time.time()

    for step in range(1, int(args.train_steps) + 1):
        pick = torch.randint(0, total_rows, (batch_size,), generator=generator)
        seq_batch = surface["sequences"][pick]
        target_blocks = surface["target_blocks"][pick].to(torch.float32)
        flat_batch = surface["flat_indices"][pick]
        act_chunks = _activation_chunks(activation_bank, flat_batch, int(surface["blocks_per_row"]))
        ctx_summary = surface["activation_summary"][pick]
        family_ids = surface["old_family_orig"][pick]
        atom_ids = surface["old_atom_ids"][pick]
        residual_blocks = _selected_residual_blocks(surface["group"], flat_batch)
        soft_stage_sum = torch.zeros(batch_size, int(surface["group"].block_size), dtype=torch.float32)
        usefulness_loss = torch.zeros((), dtype=torch.float32)
        delta_norm_loss = torch.zeros((), dtype=torch.float32)
        hard_program = torch.empty_like(seq_batch)
        p_atom_mean = 0.0
        p_res_mean = 0.0
        gain_mean = 0.0
        expected_cost_mean = 0.0

        for slot_idx in range(int(surface["num_slots"])):
            pos0 = slot_idx * int(surface["phrase_len"])
            pos1 = pos0 + int(surface["phrase_len"])
            slot_ids = torch.full((batch_size,), slot_idx, dtype=torch.int64)
            out = model(
                family_ids=family_ids,
                variant_ids=atom_ids[:, slot_idx],
                slot_ids=slot_ids,
                layer_input_summary=ctx_summary,
            )
            token_dist = F.softmax(out["logits"] / float(args.phrase_temp), dim=-1)
            hard_tokens = out["logits"].argmax(dim=-1).to(torch.uint8)
            hard_program[:, pos0:pos1] = hard_tokens

            soft_slot_blocks = torch.zeros(batch_size, int(surface["group"].block_size), dtype=torch.float32)
            for pos_idx in range(int(surface["phrase_len"])):
                soft_slot_blocks = soft_slot_blocks + token_dist[:, pos_idx, :] @ surface["stage_vector_bank"][pos0 + pos_idx]
            target_slot_blocks = slot_blocks_from_tokens(
                surface["stage_vector_bank"],
                slot_idx,
                int(surface["phrase_len"]),
                seq_batch[:, pos0:pos1],
            )
            cda_output_rel = slot_output_rel_mse(act_chunks, soft_slot_blocks, target_slot_blocks)
            slot_loss, slot_stats = _soft_usefulness_loss(
                atom_output_err=cda_output_rel,
                residual_output_err=surface["old_residual_output_rel"][pick, slot_idx],
                old_cost=surface["old_cost"][pick, slot_idx],
                atom_budget=float(args.atom_output_budget),
                residual_budget=float(args.residual_output_budget),
                atom_cost=float(args.atom_cost),
                residual_cost=float(args.atom_cost + args.residual_program_cost),
                literal_cost=float(args.literal_cost),
                conf_logit=out["confidence"],
                tau=float(args.usefulness_tau),
            )
            usefulness_loss = usefulness_loss + slot_loss
            delta_norm_loss = delta_norm_loss + out["delta"].square().mean()
            soft_stage_sum = soft_stage_sum + soft_slot_blocks
            p_atom_mean += slot_stats["p_atom_mean"]
            p_res_mean += slot_stats["p_res_mean"]
            gain_mean += slot_stats["gain_mean"]
            expected_cost_mean += slot_stats["expected_cda_cost"]

        usefulness_loss = usefulness_loss / float(surface["num_slots"])
        delta_norm_loss = delta_norm_loss / float(surface["num_slots"])
        pred_blocks = _apply_block_postprocess(surface["group"], soft_stage_sum, flat_batch, residual_blocks)
        pred_outputs = torch.einsum("nab,nb->na", act_chunks, pred_blocks)
        target_outputs = torch.einsum("nab,nb->na", act_chunks, target_blocks)
        output_num = (pred_outputs - target_outputs).square().mean(dim=1)
        output_den = target_outputs.square().mean(dim=1).clamp_min(1e-6)
        output_recon_loss = (output_num / output_den).mean()
        block_num = (pred_blocks - target_blocks).square().mean(dim=1)
        block_den = target_blocks.square().mean(dim=1).clamp_min(1e-6)
        block_aux_loss = (block_num / block_den).mean()
        loss = float(args.output_recon_weight) * output_recon_loss
        loss = loss + float(args.block_aux_weight) * block_aux_loss
        loss = loss + float(args.usefulness_weight) * usefulness_loss
        loss = loss + float(args.delta_norm_weight) * delta_norm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == int(args.train_steps):
            hard_blocks = _reconstruct_from_program_bank(
                surface["stage_vector_bank"],
                hard_program,
                surface["group"],
                flat_batch,
                residual_blocks,
            )
            hard_output_rel = _output_rel_mse_vector(
                activation_bank,
                flat_batch,
                hard_blocks,
                target_blocks,
                int(surface["blocks_per_row"]),
                int(args.assign_chunk_size),
            )
            history.append(
                {
                    "step": int(step),
                    "elapsed_s": float(time.time() - t0),
                    "loss": float(loss.item()),
                    "output_recon_loss": float(output_recon_loss.item()),
                    "block_aux_loss": float(block_aux_loss.item()),
                    "usefulness_loss": float(usefulness_loss.item()),
                    "delta_norm_loss": float(delta_norm_loss.item()),
                    "hard_token_match": float((hard_program == seq_batch).to(torch.float32).mean().item()),
                    "hard_output_rel_mse": float(hard_output_rel.mean().item()),
                    "p_atom_mean": float(p_atom_mean / float(surface["num_slots"])),
                    "p_res_mean": float(p_res_mean / float(surface["num_slots"])),
                    "gain_mean": float(gain_mean / float(surface["num_slots"])),
                    "expected_cda_cost": float(expected_cost_mean / float(surface["num_slots"])),
                }
            )
            print(
                f"[wal-cda/train] {name}: step={step}/{args.train_steps} loss={history[-1]['loss']:.4f} "
                f"out={history[-1]['output_recon_loss']:.4f} use={history[-1]['usefulness_loss']:.4f} "
                f"delta={history[-1]['delta_norm_loss']:.4f} hard_out={history[-1]['hard_output_rel_mse']:.4f}",
                flush=True,
            )

    return {
        "model": model,
        "surface": surface,
        "training_history": history,
    }


def _build_cda_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    activation_bank: torch.Tensor,
    old_state: dict[str, object],
    trained: dict[str, object],
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    model: CDAAtomGenerator = trained["model"]
    surface = _prepare_cda_surface(name, enc, activation_bank, old_state, args, sample_only=False)
    sequences = surface["sequences"]
    rows = int(surface["rows"])
    phrase_len = int(surface["phrase_len"])
    num_slots = int(surface["num_slots"])
    old_program = sequences.clone()
    cda_program = sequences.clone()
    old_atom_calls = torch.zeros(rows, dtype=torch.int32)
    old_residual_calls = torch.zeros(rows, dtype=torch.int32)
    old_literal_slots = torch.zeros(rows, dtype=torch.int32)
    cda_atom_calls = torch.zeros(rows, dtype=torch.int32)
    cda_residual_calls = torch.zeros(rows, dtype=torch.int32)
    cda_literal_slots = torch.zeros(rows, dtype=torch.int32)
    old_program_units = torch.ones(rows, dtype=torch.float32)
    cda_program_units = torch.ones(rows, dtype=torch.float32)
    cda_confidence_sum = 0.0
    cda_delta_norm_sum = 0.0
    cda_accept_output_rel_sum = 0.0
    cda_accept_count = 0
    family_counts = torch.zeros(len(old_state["family_names"]), dtype=torch.int64)

    atom_cost = float(args.atom_cost)
    residual_cost = float(args.atom_cost + args.residual_program_cost)
    literal_cost = float(args.literal_cost)

    model.eval()
    with torch.no_grad():
        for start in range(0, rows, int(args.assign_chunk_size)):
            end = min(start + int(args.assign_chunk_size), rows)
            family_orig = surface["old_family_orig"][start:end]
            atom_ids = surface["old_atom_ids"][start:end]
            semantic_family = surface["semantic_family_ids"][start:end]
            family_counts += torch.bincount(semantic_family, minlength=len(old_state["family_names"]))
            act_chunks = _activation_chunks(activation_bank, surface["flat_indices"][start:end], int(surface["blocks_per_row"]))
            ctx_summary = surface["activation_summary"][start:end]
            for slot_idx in range(num_slots):
                pos0 = slot_idx * phrase_len
                pos1 = pos0 + phrase_len
                slot_ids = torch.full((end - start,), slot_idx, dtype=torch.int64)
                out = model(
                    family_ids=family_orig,
                    variant_ids=atom_ids[:, slot_idx],
                    slot_ids=slot_ids,
                    layer_input_summary=ctx_summary,
                )
                cda_tokens = out["logits"].argmax(dim=-1).to(torch.uint8)
                target_tokens = sequences[start:end, pos0:pos1]
                target_slot_blocks = slot_blocks_from_tokens(surface["stage_vector_bank"], slot_idx, phrase_len, target_tokens)
                cda_slot_blocks = slot_blocks_from_tokens(surface["stage_vector_bank"], slot_idx, phrase_len, cda_tokens)
                cda_block_rel = slot_block_rel_mse(cda_slot_blocks, target_slot_blocks)
                cda_output_rel = slot_output_rel_mse(act_chunks, cda_slot_blocks, target_slot_blocks)

                old_atom_accept = surface["old_atom_output_rel"][start:end, slot_idx] <= float(args.atom_output_budget)
                if args.atom_block_budget is not None:
                    old_atom_accept &= surface["old_atom_block_rel"][start:end, slot_idx] <= float(args.atom_block_budget)
                old_residual_accept = (~old_atom_accept) & (surface["old_residual_output_rel"][start:end, slot_idx] <= float(args.residual_output_budget))
                if args.residual_block_budget is not None:
                    old_residual_accept &= surface["old_residual_block_rel"][start:end, slot_idx] <= float(args.residual_block_budget)
                old_literal = ~(old_atom_accept | old_residual_accept)
                if bool(old_atom_accept.any()):
                    old_program[start:end, pos0:pos1][old_atom_accept] = surface["old_approx_program"][start:end, pos0:pos1][old_atom_accept]
                if bool(old_residual_accept.any()):
                    old_program[start:end, pos0:pos1][old_residual_accept] = surface["residual_program"][start:end, pos0:pos1][old_residual_accept]
                old_atom_calls[start:end] += old_atom_accept.to(torch.int32)
                old_residual_calls[start:end] += old_residual_accept.to(torch.int32)
                old_literal_slots[start:end] += old_literal.to(torch.int32)
                old_program_units[start:end] += old_atom_accept.to(torch.float32) * atom_cost
                old_program_units[start:end] += old_residual_accept.to(torch.float32) * residual_cost
                old_program_units[start:end] += old_literal.to(torch.float32) * literal_cost

                cda_atom_accept = cda_output_rel <= float(args.atom_output_budget)
                if args.atom_block_budget is not None:
                    cda_atom_accept &= cda_block_rel <= float(args.atom_block_budget)
                cda_residual_accept = (~cda_atom_accept) & (surface["old_residual_output_rel"][start:end, slot_idx] <= float(args.residual_output_budget))
                if args.residual_block_budget is not None:
                    cda_residual_accept &= surface["old_residual_block_rel"][start:end, slot_idx] <= float(args.residual_block_budget)
                cda_literal = ~(cda_atom_accept | cda_residual_accept)
                if bool(cda_atom_accept.any()):
                    cda_program[start:end, pos0:pos1][cda_atom_accept] = cda_tokens[cda_atom_accept]
                if bool(cda_residual_accept.any()):
                    cda_program[start:end, pos0:pos1][cda_residual_accept] = surface["residual_program"][start:end, pos0:pos1][cda_residual_accept]
                cda_atom_calls[start:end] += cda_atom_accept.to(torch.int32)
                cda_residual_calls[start:end] += cda_residual_accept.to(torch.int32)
                cda_literal_slots[start:end] += cda_literal.to(torch.int32)
                cda_program_units[start:end] += cda_atom_accept.to(torch.float32) * atom_cost
                cda_program_units[start:end] += cda_residual_accept.to(torch.float32) * residual_cost
                cda_program_units[start:end] += cda_literal.to(torch.float32) * literal_cost
                cda_confidence_sum += float(torch.sigmoid(out["confidence"]).mean().item())
                cda_delta_norm_sum += float(out["delta"].square().mean().item())
                if bool(cda_atom_accept.any()):
                    cda_accept_output_rel_sum += float(cda_output_rel[cda_atom_accept].sum().item())
                    cda_accept_count += int(cda_atom_accept.sum().item())

    def build_stats(label: str, program: torch.Tensor, atom_calls: torch.Tensor, residual_calls: torch.Tensor, literal_slots: torch.Tensor, program_units: torch.Tensor) -> tuple[GroupedBlockRVQEncoding, dict[str, object]]:
        single = _clone_group_with_program_matrix(surface["group"], program)
        enc_out = GroupedBlockRVQEncoding(groups=(single,), row_slices=enc.row_slices, original_shape=enc.original_shape)
        residual_blocks = _selected_residual_blocks(surface["group"], surface["flat_indices"])
        pred_blocks = _reconstruct_from_program_bank(surface["stage_vector_bank"], program, surface["group"], surface["flat_indices"], residual_blocks)
        output_rel = _output_rel_mse_vector(
            activation_bank,
            surface["flat_indices"],
            pred_blocks,
            surface["target_blocks"],
            int(surface["blocks_per_row"]),
            int(args.assign_chunk_size),
        )
        num_tokens = rows * int(surface["raw_len"])
        stats = {
            "name": name,
            "label": label,
            "raw_program_length": int(surface["raw_len"]),
            "avg_program_length": float(program_units.mean().item()),
            "avg_low_level_calls": float(atom_calls.to(torch.float32).mean().item()),
            "avg_residual_calls": float(residual_calls.to(torch.float32).mean().item()),
            "avg_literal_slots": float(literal_slots.to(torch.float32).mean().item()),
            "accepted_budget_token_coverage": float(int((atom_calls + residual_calls).sum().item()) * phrase_len / max(num_tokens, 1)),
            "budgeted_output_rel_mse": float(output_rel.mean().item()),
            "family_entropy": float(
                -(
                    (family_counts.to(torch.float64) / max(rows, 1)).clamp_min(1e-12)
                    * (family_counts.to(torch.float64) / max(rows, 1)).clamp_min(1e-12).log()
                ).sum().item()
            ),
        }
        return enc_out, stats

    old_budgeted_enc, old_stats = build_stats("budgeted_exact_old", old_program, old_atom_calls, old_residual_calls, old_literal_slots, old_program_units)
    cda_budgeted_enc, cda_stats = build_stats("budgeted_exact_cda", cda_program, cda_atom_calls, cda_residual_calls, cda_literal_slots, cda_program_units)
    cda_stats["mean_confidence"] = float(cda_confidence_sum / max(rows * num_slots, 1))
    cda_stats["mean_delta_norm"] = float(cda_delta_norm_sum / max(rows * num_slots, 1))
    cda_stats["accepted_cda_output_rel_mse"] = float(cda_accept_output_rel_sum / max(cda_accept_count, 1))
    artifact = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": {
            "ctx_dim": int(args.ctx_dim),
            "cda_hidden_dim": int(args.cda_hidden_dim),
            "rank": int(args.rank),
            "summary_dim": int(args.summary_dim),
            "delta_clip": float(args.delta_clip),
            "phrase_temp": float(args.phrase_temp),
            "usefulness_tau": float(args.usefulness_tau),
        },
        "training_history": trained["training_history"],
        "base_family_names": list(old_state["family_names"]),
    }
    return old_budgeted_enc, old_stats, cda_budgeted_enc, cda_stats, artifact


def main() -> None:
    args = _parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    print(f"[activations] capture {args.lha_calibration_windows} windows for {len(TARGETS)} targets", flush=True)
    layer_inputs = _collect_layer_inputs_budgeted(
        ids,
        int(args.lha_calibration_windows),
        TARGETS,
        int(args.activation_rows),
        int(args.capture_max_len),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    activation_banks = {name: _build_activation_bank(layer_inputs[name], int(args.block_size)) for name in TARGETS}
    old_states = _load_or_build_old_basis(Path(args.sbc_artifact), current_enc, activation_banks, args)

    old_budgeted_encodings = {}
    cda_budgeted_encodings = {}
    old_rows = []
    cda_rows = []
    old_compare = []
    cda_compare = []
    artifacts = {}
    for name in TARGETS:
        print(f"[wal-cda] train constrained adapter for {name}", flush=True)
        trained = _train_cda_adapter(name, current_enc[name], activation_banks[name], old_states[name], args)
        old_enc, old_stats, cda_enc, cda_stats, artifact = _build_cda_layer(
            name,
            current_enc[name],
            activation_banks[name],
            old_states[name],
            trained,
            args,
        )
        old_budgeted_encodings[name] = old_enc
        cda_budgeted_encodings[name] = cda_enc
        old_rows.append(old_stats)
        cda_rows.append(cda_stats)
        old_compare.append({"name": name, **_compare_encodings(current_enc[name], old_enc)})
        cda_compare.append({"name": name, **_compare_encodings(current_enc[name], cda_enc)})
        artifacts[name] = artifact
        print(
            f"[wal-cda] {name}: old_avg_program={old_stats['avg_program_length']:.3f}/{old_stats['raw_program_length']} "
            f"cda_avg_program={cda_stats['avg_program_length']:.3f}/{cda_stats['raw_program_length']} "
            f"cda_atom_calls={cda_stats['avg_low_level_calls']:.3f} residual_calls={cda_stats['avg_residual_calls']:.3f} "
            f"cda_rel={cda_stats['budgeted_output_rel_mse']:.4f}",
            flush=True,
        )

    old_budgeted_cache = Path(args.old_budgeted_cache)
    cda_budgeted_cache = Path(args.cda_budgeted_cache)
    save_grouped_encoding_map(old_budgeted_cache, old_budgeted_encodings)
    save_grouped_encoding_map(cda_budgeted_cache, cda_budgeted_encodings)
    artifact_path = Path(args.cda_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "targets": list(TARGETS), "artifacts": artifacts}, artifact_path)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] strict legacy via {args.matmul_strategy}", flush=True)
    legacy_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] old budgeted basis via {args.matmul_strategy}", flush=True)
    old_budgeted_eval = _run_preencoded_eval(ids, old_budgeted_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] CDA budgeted basis via {args.matmul_strategy}", flush=True)
    cda_budgeted_eval = _run_preencoded_eval(ids, cda_budgeted_cache, args.matmul_strategy, args.num_windows)

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "lha_calibration_windows": int(args.lha_calibration_windows),
        "activation_rows": int(args.activation_rows),
        "matmul_strategy": args.matmul_strategy,
        "atom_block_budget": None if args.atom_block_budget is None else float(args.atom_block_budget),
        "atom_output_budget": float(args.atom_output_budget),
        "residual_block_budget": None if args.residual_block_budget is None else float(args.residual_block_budget),
        "residual_output_budget": float(args.residual_output_budget),
        "residual_program_cost": float(args.residual_program_cost),
        "dense": dense,
        "current_cache": str(current_cache),
        "old_budgeted_cache": str(old_budgeted_cache),
        "cda_budgeted_cache": str(cda_budgeted_cache),
        "cda_artifact": str(artifact_path),
        "old_budgeted": old_rows,
        "cda_budgeted": cda_rows,
        "old_budgeted_compare": old_compare,
        "cda_budgeted_compare": cda_compare,
        "legacy_eval": legacy_eval,
        "old_budgeted_eval": old_budgeted_eval,
        "cda_budgeted_eval": cda_budgeted_eval,
        "delta_old_vs_legacy": {
            "ppl_delta": float(old_budgeted_eval["metrics"]["perplexity"] - legacy_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(old_budgeted_eval["metrics"]["tok_s"] - legacy_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(old_budgeted_eval["eval_peak_mb"] - legacy_eval["eval_peak_mb"]),
        },
        "delta_cda_vs_legacy": {
            "ppl_delta": float(cda_budgeted_eval["metrics"]["perplexity"] - legacy_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(cda_budgeted_eval["metrics"]["tok_s"] - legacy_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(cda_budgeted_eval["eval_peak_mb"] - legacy_eval["eval_peak_mb"]),
        },
        "delta_cda_vs_old": {
            "ppl_delta": float(cda_budgeted_eval["metrics"]["perplexity"] - old_budgeted_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(cda_budgeted_eval["metrics"]["tok_s"] - old_budgeted_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(cda_budgeted_eval["eval_peak_mb"] - old_budgeted_eval["eval_peak_mb"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(f"  dense:               ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}", flush=True)
    print(f"  strict_legacy:       ppl={legacy_eval['metrics']['perplexity']:.4f} tok/s={legacy_eval['metrics']['tok_s']:.2f} peak_mb={legacy_eval['eval_peak_mb']:.1f}", flush=True)
    print(f"  budgeted_exact_old:  ppl={old_budgeted_eval['metrics']['perplexity']:.4f} tok/s={old_budgeted_eval['metrics']['tok_s']:.2f} peak_mb={old_budgeted_eval['eval_peak_mb']:.1f}", flush=True)
    print(f"  budgeted_exact_cda:  ppl={cda_budgeted_eval['metrics']['perplexity']:.4f} tok/s={cda_budgeted_eval['metrics']['tok_s']:.2f} peak_mb={cda_budgeted_eval['eval_peak_mb']:.1f}", flush=True)
    for old_row, cda_row in zip(old_rows, cda_rows):
        print(
            f"  {old_row['name']}: old_program={old_row['avg_program_length']:.3f}/{old_row['raw_program_length']} "
            f"cda_program={cda_row['avg_program_length']:.3f}/{cda_row['raw_program_length']} "
            f"cda_atom_calls={cda_row['avg_low_level_calls']:.3f} residual_calls={cda_row['avg_residual_calls']:.3f} "
            f"conf={cda_row['mean_confidence']:.3f} delta_norm={cda_row['mean_delta_norm']:.4f}",
            flush=True,
        )
    print(f"  delta(old-legacy): ppl={result['delta_old_vs_legacy']['ppl_delta']:+.6f} tok/s={result['delta_old_vs_legacy']['tok_s_delta']:+.2f} peak_mb={result['delta_old_vs_legacy']['peak_mb_delta']:+.1f}", flush=True)
    print(f"  delta(cda-legacy): ppl={result['delta_cda_vs_legacy']['ppl_delta']:+.6f} tok/s={result['delta_cda_vs_legacy']['tok_s_delta']:+.2f} peak_mb={result['delta_cda_vs_legacy']['peak_mb_delta']:+.1f}", flush=True)
    print(f"  delta(cda-old):    ppl={result['delta_cda_vs_old']['ppl_delta']:+.6f} tok/s={result['delta_cda_vs_old']['tok_s_delta']:+.2f} peak_mb={result['delta_cda_vs_old']['peak_mb_delta']:+.1f}", flush=True)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()