"""M27 WAL-LDI Step 10a: learned discrete ISA with semantic constraints.

This probe stops trying to wrap the current flat language from above and instead
builds a small two-level ISA directly on the block-program surface.

For each target layer:
  - learn a small set of semantic high-level instructions (families)
  - constrain family assignment with both semantic features and token-sequence
    consistency
  - learn low-level slot atoms conditioned on each high-level family

Two surfaces are measured:
  1. exact WAL-LDI: FAMILY + low-level atoms + literal corrections
  2. ldi_only: FAMILY-conditioned low-level atoms only, no corrections
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import BlockRVQEncoding, GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
from dwl2_dynamic_route.src.encoding_io import load_grouped_encoding_map, save_grouped_encoding_map
from dwl2_dynamic_route.src.runtime import replace_with_preencoded_packed_block_rvq


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512
TARGETS = (
    "model.layers.54.mlp.gate_proj",
    "model.layers.54.mlp.up_proj",
)


def _eval_ids(tok: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


def _load_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def _reset_peaks() -> None:
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)


def _vram() -> list[dict[str, float]]:
    return [
        {
            "device": idx,
            "allocated_mb": round(torch.cuda.memory_allocated(idx) / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(idx) / 2**20, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated(idx) / 2**20, 1),
        }
        for idx in range(torch.cuda.device_count())
    ]


def _peak(snap: list[dict[str, float]]) -> float:
    return max((float(item["peak_mb"]) for item in snap), default=0.0)


def _evaluate(model: AutoModelForCausalLM, ids: torch.Tensor, n_windows: int) -> dict[str, float]:
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls: list[float] = []
    total_tokens = 0
    prev_end = 0
    total_len = ids.size(1)
    n_windows = min(n_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    t0 = time.time()
    with torch.no_grad():
        for idx in range(n_windows):
            begin = idx * STRIDE
            end = min(begin + MAX_LEN, total_len)
            target_len = end - prev_end if idx > 0 else end - begin
            chunk = ids[:, begin:end].to(device)
            target = chunk.clone()
            if idx > 0:
                target[:, :-target_len] = -100
            loss = model(chunk, labels=target).loss.item()
            nlls.append(loss * target_len)
            total_tokens += target_len
            prev_end = end
            if (idx + 1) % 8 == 0 or idx + 1 == n_windows:
                dt = time.time() - t0
                ppl = math.exp(sum(nlls) / total_tokens)
                print(f"  {idx + 1}/{n_windows}  ppl={ppl:.4f}  tok/s={total_tokens / max(dt, 1e-9):.1f}", flush=True)
    dt = time.time() - t0
    return {
        "perplexity": math.exp(sum(nlls) / total_tokens),
        "total_tokens": total_tokens,
        "num_windows": n_windows,
        "elapsed_s": dt,
        "tok_s": total_tokens / max(dt, 1e-9),
    }


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).square().mean()
    den = a.float().square().mean().clamp_min(1e-12)
    return float((num / den).item())


def _load_cached_subset(path: Path, targets: tuple[str, ...]) -> dict[str, GroupedBlockRVQEncoding]:
    if not path.exists():
        return {}
    loaded = load_grouped_encoding_map(path)
    return {name: enc for name, enc in loaded.items() if name in targets}


def _build_current_cache(args: argparse.Namespace, cache_path: Path, targets: tuple[str, ...]) -> dict[str, GroupedBlockRVQEncoding]:
    encodings = _load_cached_subset(cache_path, targets)
    if not encodings and args.bootstrap_cache:
        encodings.update(_load_cached_subset(Path(args.bootstrap_cache), targets))
    missing = [name for name in targets if name not in encodings]
    if not missing and cache_path.exists() and not args.rebuild_cache:
        print(f"[cache] load {cache_path}", flush=True)
        return encodings
    if not missing and not args.rebuild_cache:
        print(f"[cache] reuse bootstrap subset -> {cache_path}", flush=True)
        save_grouped_encoding_map(cache_path, encodings)
        return encodings
    print(f"[cache] build/extend {cache_path}", flush=True)
    model = _load_model()
    module_map = dict(model.named_modules())
    for name in missing:
        module = module_map[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name} is not nn.Linear")
        print(f"  encode {name}", flush=True)
        encodings[name] = encode_grouped_block_residual_vq(
            module.weight.detach(),
            group_rows=args.group_rows,
            block_size=args.block_size,
            codebook_size=args.codebook_size,
            num_stages=args.num_stages,
            product_splits=args.product_splits,
            calibrate_stage_scales=args.calibrate_stage_scales,
            sample_limit=args.sample_limit,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
    save_grouped_encoding_map(cache_path, encodings)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encodings


def _flatten_group_sequences(group: BlockRVQEncoding) -> torch.Tensor:
    return torch.stack([ids.reshape(-1).to(torch.uint8).cpu() for ids in group.stage_ids], dim=1).contiguous()


def _sample_training_sequences(enc: GroupedBlockRVQEncoding, max_rows: int, seed: int) -> torch.Tensor:
    total_blocks = sum(int(group.stage_ids[0].numel()) for group in enc.groups)
    if total_blocks <= max_rows:
        return torch.cat([_flatten_group_sequences(group) for group in enc.groups], dim=0)
    samples = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for group in enc.groups:
        group_blocks = int(group.stage_ids[0].numel())
        budget = max(1, int(round(max_rows * group_blocks / total_blocks)))
        seq_mat = _flatten_group_sequences(group)
        budget = min(budget, int(seq_mat.shape[0]))
        pick = torch.randperm(int(seq_mat.shape[0]), generator=generator)[:budget]
        samples.append(seq_mat[pick])
    return torch.cat(samples, dim=0)[:max_rows].contiguous()


def _nearest_phrases(slot_sequences: torch.Tensor, phrases: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = int(slot_sequences.shape[0])
    assignments = torch.empty(total_rows, dtype=torch.int64)
    distances = torch.empty(total_rows, dtype=torch.int16)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = slot_sequences[start:end]
        dist = (chunk[:, None, :] != phrases[None, :, :]).sum(dim=2)
        best_dist, best_idx = dist.min(dim=1)
        assignments[start:end] = best_idx.to(torch.int64)
        distances[start:end] = best_dist.to(torch.int16)
    return assignments, distances


def _learn_slot_phrases(
    slot_sequences: torch.Tensor,
    num_variants: int,
    base: int,
    iters: int,
    assign_chunk_size: int,
) -> dict[str, object]:
    train_rows = int(slot_sequences.shape[0])
    if train_rows < 1:
        raise ValueError("slot_sequences must be non-empty")
    if train_rows < num_variants:
        pad = slot_sequences[torch.arange(num_variants) % train_rows]
        slot_sequences = pad.contiguous()
        train_rows = int(slot_sequences.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(train_rows, generator=generator)
    phrases = slot_sequences[perm[:num_variants]].clone()
    history = []
    for iter_idx in range(iters):
        assignments, distances = _nearest_phrases(slot_sequences, phrases, assign_chunk_size)
        counts = torch.bincount(assignments, minlength=num_variants)
        updated = phrases.clone()
        for pos in range(slot_sequences.shape[1]):
            flat = assignments * base + slot_sequences[:, pos].to(torch.int64)
            hist = torch.bincount(flat, minlength=num_variants * base).view(num_variants, base)
            updated[:, pos] = hist.argmax(dim=1).to(torch.uint8)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            reseed = torch.randint(0, train_rows, (empty.numel(),), generator=generator)
            updated[empty] = slot_sequences[reseed]
        phrases = updated.contiguous()
        history.append(
            {
                "iter": int(iter_idx + 1),
                "mean_hamming": float(distances.to(torch.float32).mean().item()),
                "p50_hamming": float(torch.quantile(distances.to(torch.float32), 0.50).item()),
                "p90_hamming": float(torch.quantile(distances.to(torch.float32), 0.90).item()),
            }
        )
    final_assignments, final_distances = _nearest_phrases(slot_sequences, phrases, assign_chunk_size)
    final_counts = torch.bincount(final_assignments, minlength=num_variants)
    return {
        "phrases": phrases,
        "train_counts": final_counts,
        "train_mean_hamming": float(final_distances.to(torch.float32).mean().item()),
        "history": history,
    }


def _clone_group_with_program_matrix(group: BlockRVQEncoding, program_matrix: torch.Tensor) -> BlockRVQEncoding:
    shape = group.stage_ids[0].shape
    stage_ids = []
    for stage_idx, ids in enumerate(group.stage_ids):
        stage_ids.append(program_matrix[:, stage_idx].reshape(shape).to(dtype=ids.dtype))
    return BlockRVQEncoding(
        stage_ids=tuple(stage_ids),
        codebooks=group.codebooks,
        stage_value_dims=group.stage_value_dims,
        stages_per_split=group.stages_per_split,
        stage_scales=group.stage_scales,
        residual_correction=group.residual_correction,
        residual_signs=group.residual_signs,
        residual_scale=group.residual_scale,
        row_scale=group.row_scale,
        block_scale=group.block_scale,
        transform_kind=group.transform_kind,
        transform_matrix=group.transform_matrix,
        transform_bias=group.transform_bias,
        product_splits=group.product_splits,
        original_shape=group.original_shape,
        padded_cols=group.padded_cols,
        block_size=group.block_size,
        sample_rel_mse=group.sample_rel_mse,
    )


def _role_name(layer_name: str) -> str:
    if layer_name.endswith("gate_proj"):
        return "MLP_GATE"
    if layer_name.endswith("up_proj"):
        return "MLP_UP"
    raise ValueError(f"unsupported target for WAL-LDI: {layer_name}")


def _family_names_for_role(role_name: str, num_families: int) -> tuple[str, ...]:
    if num_families == 4:
        suffixes = ("COMMON_CORE", "COMMON_MIXED", "RARE_STRUCTURED", "OUTLIER_DIFFUSE")
        return tuple(f"{role_name}_{suffix}" for suffix in suffixes)
    return tuple(f"{role_name}_FAMILY_{idx:02d}" for idx in range(num_families))


def _compute_token_model(enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    sample_sequences = _sample_training_sequences(enc, int(args.train_samples), seed=0)
    raw_len = int(sample_sequences.shape[1])
    base = max(int(codebook.shape[0]) for group in enc.groups for codebook in group.codebooks)
    counts = torch.zeros(raw_len, base, dtype=torch.int64)
    for pos in range(raw_len):
        counts[pos] = torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base)
    probs = (counts.to(torch.float32) + 1.0) / (float(sample_sequences.shape[0]) + float(base))
    stage_surprisal = (-torch.log(probs)).contiguous()
    modal_tokens = counts.argmax(dim=1).to(torch.uint8)
    return {
        "sample_sequences": sample_sequences,
        "raw_len": raw_len,
        "base": base,
        "stage_surprisal": stage_surprisal,
        "modal_tokens": modal_tokens,
    }


def _semantic_features(
    sequence_matrix: torch.Tensor,
    stage_surprisal: torch.Tensor,
    modal_tokens: torch.Tensor,
    phrase_len: int,
) -> torch.Tensor:
    rows = int(sequence_matrix.shape[0])
    raw_len = int(sequence_matrix.shape[1])
    num_slots = raw_len // phrase_len
    avg_surprisal = torch.zeros(rows, dtype=torch.float32)
    modal_match = torch.zeros(rows, dtype=torch.float32)
    slot_surprisal = torch.zeros(rows, num_slots, dtype=torch.float32)
    slot_modal = torch.zeros(rows, num_slots, dtype=torch.float32)
    for pos in range(raw_len):
        ids = sequence_matrix[:, pos].to(torch.int64)
        surprisal = stage_surprisal[pos][ids]
        avg_surprisal += surprisal
        slot_idx = pos // phrase_len
        slot_surprisal[:, slot_idx] += surprisal
        match = (sequence_matrix[:, pos] == modal_tokens[pos]).to(torch.float32)
        modal_match += match
        slot_modal[:, slot_idx] += match
    avg_surprisal /= float(raw_len)
    modal_match /= float(raw_len)
    slot_surprisal /= float(phrase_len)
    slot_modal /= float(phrase_len)
    return torch.cat(
        [avg_surprisal.unsqueeze(1), modal_match.unsqueeze(1), slot_surprisal, slot_modal],
        dim=1,
    ).contiguous()


def _combined_family_distance(
    norm_features: torch.Tensor,
    sequence_matrix: torch.Tensor,
    feature_centroids: torch.Tensor,
    sequence_centroids: torch.Tensor,
    semantic_weight: float,
    sequence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_dist = (norm_features[:, None, :] - feature_centroids[None, :, :]).square().mean(dim=2)
    sequence_dist = (sequence_matrix[:, None, :] != sequence_centroids[None, :, :]).to(torch.float32).mean(dim=2)
    combined = semantic_weight * feature_dist + sequence_weight * sequence_dist
    return combined, feature_dist, sequence_dist


def _learn_semantic_families(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    token_model = _compute_token_model(enc, args)
    sample_sequences = token_model["sample_sequences"]
    raw_features = _semantic_features(
        sample_sequences,
        token_model["stage_surprisal"],
        token_model["modal_tokens"],
        int(args.phrase_len),
    )
    feature_mean = raw_features.mean(dim=0)
    feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    norm_features = ((raw_features - feature_mean) / feature_std).contiguous()

    num_families = int(args.num_families)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    perm = torch.randperm(int(sample_sequences.shape[0]), generator=generator)
    feature_centroids = norm_features[perm[:num_families]].clone()
    sequence_centroids = sample_sequences[perm[:num_families]].clone()
    history = []
    semantic_weight = float(args.semantic_weight)
    sequence_weight = float(args.sequence_weight)

    for iter_idx in range(int(args.family_iters)):
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
                family_sequences = sample_sequences[mask]
                for pos in range(int(sample_sequences.shape[1])):
                    hist = torch.bincount(family_sequences[:, pos].to(torch.int64), minlength=int(token_model["base"]))
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
    best_dist, assignments = combined.min(dim=1)
    counts = torch.bincount(assignments, minlength=num_families)
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

    family_summaries = []
    for family_idx, family_name in enumerate(family_names):
        mask = assignments == family_idx
        count = int(mask.sum().item())
        if count > 0:
            own_feature = feature_dist[:, permute[family_idx]][mask]
            own_sequence = sequence_dist[:, permute[family_idx]][mask]
            own_combined = combined[:, permute[family_idx]][mask]
            other_cols = [int(permute[idx].item()) for idx in range(num_families) if idx != family_idx]
            other = combined[mask][:, other_cols].min(dim=1).values if other_cols else own_combined
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "sample_count": count,
                    "sample_share": float(count / max(int(sample_sequences.shape[0]), 1)),
                    "sample_avg_surprisal": float(raw_features[mask, 0].mean().item()),
                    "sample_modal_match": float(raw_features[mask, 1].mean().item()),
                    "sample_feature_radius": float(own_feature.mean().item()),
                    "sample_sequence_radius": float(own_sequence.mean().item()),
                    "sample_margin": float((other - own_combined).mean().item()),
                }
            )
        else:
            family_summaries.append(
                {
                    "family_id": int(family_idx),
                    "family_name": family_name,
                    "sample_count": 0,
                    "sample_share": 0.0,
                    "sample_avg_surprisal": 0.0,
                    "sample_modal_match": 0.0,
                    "sample_feature_radius": 0.0,
                    "sample_sequence_radius": 0.0,
                    "sample_margin": 0.0,
                }
            )

    token_model.update(
        {
            "raw_features": raw_features,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_centroids": feature_centroids,
            "sequence_centroids": sequence_centroids,
            "sample_family_ids": assignments,
            "family_names": family_names,
            "family_summaries": family_summaries,
            "semantic_weight": semantic_weight,
            "sequence_weight": sequence_weight,
            "family_history": history,
        }
    )
    return token_model


def _assign_families(sequence_matrix: torch.Tensor, family_model: dict[str, object], chunk_size: int) -> dict[str, object]:
    total_rows = int(sequence_matrix.shape[0])
    num_families = len(family_model["family_names"])
    family_ids = torch.empty(total_rows, dtype=torch.int64)
    family_count_totals = torch.zeros(num_families, dtype=torch.int64)
    family_surprisal_sum = torch.zeros(num_families, dtype=torch.float64)
    family_modal_match_sum = torch.zeros(num_families, dtype=torch.float64)
    family_feature_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_sequence_radius_sum = torch.zeros(num_families, dtype=torch.float64)
    family_margin_sum = torch.zeros(num_families, dtype=torch.float64)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = sequence_matrix[start:end]
        raw_features = _semantic_features(
            chunk,
            family_model["stage_surprisal"],
            family_model["modal_tokens"],
            int(chunk.shape[1]) // 4,
        )
        norm_features = ((raw_features - family_model["feature_mean"]) / family_model["feature_std"]).contiguous()
        combined, feature_dist, sequence_dist = _combined_family_distance(
            norm_features,
            chunk,
            family_model["feature_centroids"],
            family_model["sequence_centroids"],
            float(family_model["semantic_weight"]),
            float(family_model["sequence_weight"]),
        )
        best_dist, best_ids = combined.min(dim=1)
        family_ids[start:end] = best_ids.to(torch.int64)
        for family_idx in range(num_families):
            mask = best_ids == family_idx
            if not bool(mask.any()):
                continue
            own_combined = best_dist[mask]
            other_cols = [idx for idx in range(num_families) if idx != family_idx]
            other = combined[mask][:, other_cols].min(dim=1).values if other_cols else own_combined
            family_count_totals[family_idx] += int(mask.sum().item())
            family_surprisal_sum[family_idx] += float(raw_features[mask, 0].sum().item())
            family_modal_match_sum[family_idx] += float(raw_features[mask, 1].sum().item())
            family_feature_radius_sum[family_idx] += float(feature_dist[mask, family_idx].sum().item())
            family_sequence_radius_sum[family_idx] += float(sequence_dist[mask, family_idx].sum().item())
            family_margin_sum[family_idx] += float((other - own_combined).sum().item())
    return {
        "family_ids": family_ids,
        "family_count_totals": family_count_totals,
        "family_surprisal_sum": family_surprisal_sum,
        "family_modal_match_sum": family_modal_match_sum,
        "family_feature_radius_sum": family_feature_radius_sum,
        "family_sequence_radius_sum": family_sequence_radius_sum,
        "family_margin_sum": family_margin_sum,
    }


def _instruction_name(family_name: str, slot_idx: int, variant_idx: int) -> str:
    return f"{family_name}_SLOT_{slot_idx}_ATOM_{variant_idx}"


def _compare_encodings(original: GroupedBlockRVQEncoding, rebuilt: GroupedBlockRVQEncoding) -> dict[str, object]:
    if original is rebuilt:
        return {
            "exact_stage_ids": True,
            "max_stage_id_diff": 0,
            "recon_rel_mse": 0.0,
        }
    seq_equal = True
    max_id_diff = 0
    for g_a, g_b in zip(original.groups, rebuilt.groups):
        for ids_a, ids_b in zip(g_a.stage_ids, g_b.stage_ids):
            ids_a_cpu = ids_a.to(torch.int64).cpu()
            ids_b_cpu = ids_b.to(torch.int64).cpu()
            diff = (ids_a_cpu - ids_b_cpu).abs().max().item()
            max_id_diff = max(max_id_diff, int(diff))
            if not torch.equal(ids_a_cpu, ids_b_cpu):
                seq_equal = False
    recon_a = original.reconstruct(out_dtype=torch.bfloat16).cpu()
    recon_b = rebuilt.reconstruct(out_dtype=torch.bfloat16).cpu()
    return {
        "exact_stage_ids": bool(seq_equal),
        "max_stage_id_diff": int(max_id_diff),
        "recon_rel_mse": _rel_mse(recon_a, recon_b),
    }


def _build_ldi_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    family_model = _learn_semantic_families(name, enc, args)
    raw_len = int(family_model["raw_len"])
    phrase_len = int(args.phrase_len)
    if raw_len % phrase_len != 0:
        raise ValueError("phrase_len must divide raw program length")
    num_slots = raw_len // phrase_len
    if num_slots != 4:
        raise ValueError("the first WAL-LDI prototype expects exactly 4 phrase slots")

    sample_sequences = family_model["sample_sequences"]
    sample_family_ids = family_model["sample_family_ids"]
    family_names = family_model["family_names"]
    base = int(family_model["base"])
    slot_learned: list[list[dict[str, object]]] = []
    slot_summaries = []
    for family_idx, family_name in enumerate(family_names):
        family_mask = sample_family_ids == family_idx
        family_sequences = sample_sequences[family_mask]
        print(
            f"[wal-ldi/train] {name}: family={family_name} samples={family_sequences.shape[0]} variants={args.num_slot_variants}",
            flush=True,
        )
        family_slots = []
        family_slot_summaries = []
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            learned = _learn_slot_phrases(
                family_sequences[:, start:end],
                int(args.num_slot_variants),
                base,
                int(args.atom_iters),
                int(args.assign_chunk_size),
            )
            family_slots.append(learned)
            variants = []
            total = int(learned["train_counts"].sum().item())
            for variant_idx in range(int(args.num_slot_variants)):
                variants.append(
                    {
                        "variant_id": int(variant_idx),
                        "instruction_name": _instruction_name(family_name, slot_idx, variant_idx),
                        "train_count": int(learned["train_counts"][variant_idx].item()),
                        "phrase": learned["phrases"][variant_idx].to(torch.int64).tolist(),
                    }
                )
            family_slot_summaries.append(
                {
                    "slot_index": int(slot_idx),
                    "train_total": int(total),
                    "train_mean_hamming": float(learned["train_mean_hamming"]),
                    "history": learned["history"],
                    "variants": variants,
                }
            )
        slot_learned.append(family_slots)
        slot_summaries.append(
            {
                "family_id": int(family_idx),
                "family_name": family_name,
                "slots": family_slot_summaries,
            }
        )

    total_blocks = 0
    total_program_units = 0
    total_high_level_calls = 0
    total_low_level_calls = 0
    total_literal_corrections = 0
    total_low_level_tokens = 0
    total_ldi_only_hamming = 0
    total_slot_assignments = 0
    family_count_totals = torch.zeros(len(family_names), dtype=torch.int64)
    family_low_level_calls = torch.zeros(len(family_names), dtype=torch.int64)
    family_literal_corrections = torch.zeros(len(family_names), dtype=torch.int64)
    family_program_units = torch.zeros(len(family_names), dtype=torch.int64)
    family_low_level_tokens = torch.zeros(len(family_names), dtype=torch.int64)
    family_token_match_hamming = torch.zeros(len(family_names), dtype=torch.int64)
    family_instruction_usage: Counter[tuple[int, int, int]] = Counter()
    approx_groups = []
    group_rows = []
    for group in enc.groups:
        seq_mat = _flatten_group_sequences(group)
        family_assign = _assign_families(seq_mat, family_model, int(args.assign_chunk_size))
        family_ids = family_assign["family_ids"]
        rows = int(seq_mat.shape[0])
        total_blocks += rows
        total_high_level_calls += rows
        total_program_units += rows
        family_count_totals += family_assign["family_count_totals"]
        approx_slots = []
        group_low_level_calls = torch.zeros(rows, dtype=torch.int32)
        group_literal_corrections = torch.zeros(rows, dtype=torch.int32)
        group_program_units = torch.ones(rows, dtype=torch.int32)
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            slot_sequences = seq_mat[:, start:end]
            approx_slot = torch.empty_like(slot_sequences)
            for family_idx in range(len(family_names)):
                mask = family_ids == family_idx
                if not bool(mask.any()):
                    continue
                learned = slot_learned[family_idx][slot_idx]
                masked_sequences = slot_sequences[mask]
                assignments, distances = _nearest_phrases(masked_sequences, learned["phrases"], int(args.assign_chunk_size))
                dist_i32 = distances.to(torch.int32)
                use_atom = (dist_i32 + 1) < phrase_len
                saved_tokens = torch.where(use_atom, phrase_len - dist_i32, torch.zeros_like(dist_i32))
                group_low_level_calls[mask] += use_atom.to(torch.int32)
                group_literal_corrections[mask] += torch.where(use_atom, dist_i32, torch.full_like(dist_i32, phrase_len))
                group_program_units[mask] += torch.where(use_atom, dist_i32 + 1, torch.full_like(dist_i32, phrase_len))
                total_low_level_tokens += int(saved_tokens.sum().item())
                family_low_level_tokens[family_idx] += int(saved_tokens.sum().item())
                total_ldi_only_hamming += int(dist_i32.sum().item())
                family_token_match_hamming[family_idx] += int(dist_i32.sum().item())
                total_slot_assignments += int(dist_i32.numel())
                if bool(use_atom.any()):
                    slot_counts = torch.bincount(assignments[use_atom], minlength=int(args.num_slot_variants))
                    for variant_idx, count in enumerate(slot_counts.tolist()):
                        if count > 0:
                            family_instruction_usage[(int(family_idx), int(slot_idx), int(variant_idx))] += int(count)
                approx_slot[mask] = learned["phrases"][assignments].to(torch.uint8)
            approx_slots.append(approx_slot)
        total_low_level_calls += int(group_low_level_calls.sum().item())
        total_literal_corrections += int(group_literal_corrections.sum().item())
        total_program_units += int(group_program_units.sum().item()) - rows
        group_rows.append(
            {
                "group_idx": int(len(group_rows)),
                "block_count": rows,
                "family_entropy": float(
                    -(
                        (family_assign["family_count_totals"].to(torch.float64) / max(rows, 1)).clamp_min(1e-12)
                        * (family_assign["family_count_totals"].to(torch.float64) / max(rows, 1)).clamp_min(1e-12).log()
                    ).sum().item()
                ),
            }
        )
        for family_idx in range(len(family_names)):
            mask = family_ids == family_idx
            if not bool(mask.any()):
                continue
            family_low_level_calls[family_idx] += int(group_low_level_calls[mask].sum().item())
            family_literal_corrections[family_idx] += int(group_literal_corrections[mask].sum().item())
            family_program_units[family_idx] += int(group_program_units[mask].sum().item())
        approx_groups.append(_clone_group_with_program_matrix(group, torch.cat(approx_slots, dim=1)))

    ldi_only = GroupedBlockRVQEncoding(
        groups=tuple(approx_groups),
        row_slices=enc.row_slices,
        original_shape=enc.original_shape,
    )

    family_summaries = []
    total_instruction_count = len(family_names) * (1 + num_slots * int(args.num_slot_variants))
    total_tokens = max(total_blocks * raw_len, 1)
    family_probs = family_count_totals.to(torch.float64) / max(total_blocks, 1)
    family_entropy = float(-(family_probs.clamp_min(1e-12) * family_probs.clamp_min(1e-12).log()).sum().item())
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
                    "block_share": float(count / max(total_blocks, 1)),
                    "avg_surprisal": float(family_assign["family_surprisal_sum"][family_idx].item() / count),
                    "avg_modal_match": float(family_assign["family_modal_match_sum"][family_idx].item() / count),
                    "avg_feature_radius": float(family_assign["family_feature_radius_sum"][family_idx].item() / count),
                    "avg_sequence_radius": float(family_assign["family_sequence_radius_sum"][family_idx].item() / count),
                    "avg_margin": float(family_assign["family_margin_sum"][family_idx].item() / count),
                    "avg_low_level_calls": float(family_low_level_calls[family_idx].item() / count),
                    "avg_literal_corrections": float(family_literal_corrections[family_idx].item() / count),
                    "avg_program_length": float(family_program_units[family_idx].item() / count),
                    "low_level_token_coverage": float(family_low_level_tokens[family_idx].item() / max(count * raw_len, 1)),
                    "ldi_only_token_match": float(1.0 - family_token_match_hamming[family_idx].item() / max(count * raw_len, 1)),
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
                    "avg_low_level_calls": 0.0,
                    "avg_literal_corrections": 0.0,
                    "avg_program_length": 0.0,
                    "low_level_token_coverage": 0.0,
                    "ldi_only_token_match": 0.0,
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
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_high_level_calls": 1.0,
        "avg_low_level_calls": float(total_low_level_calls / max(total_blocks, 1)),
        "avg_literal_corrections": float(total_literal_corrections / max(total_blocks, 1)),
        "low_level_token_coverage": float(total_low_level_tokens / total_tokens),
        "hierarchical_instruction_share": float((total_high_level_calls + total_low_level_calls) / max(total_program_units, 1)),
        "program_compression_ratio": float(total_program_units / total_tokens),
        "instruction_vocab_size": int(total_instruction_count),
        "high_level_instruction_count": int(len(family_names)),
        "low_level_instruction_count": int(len(family_names) * num_slots * int(args.num_slot_variants)),
        "active_family_count": int(sum(1 for count in family_count_totals.tolist() if count > 0)),
        "family_entropy": float(family_entropy),
        "ldi_only_avg_hamming": float(total_ldi_only_hamming / max(total_blocks, 1)),
        "ldi_only_token_match": float(1.0 - total_ldi_only_hamming / total_tokens),
        "family_history": family_model["family_history"],
        "family_summaries": family_summaries,
        "slot_summaries": slot_summaries,
        "top_atoms": top_atoms,
        "group_rows": group_rows,
    }
    artifact = {
        "family_names": family_names,
        "family_feature_centroids": family_model["feature_centroids"],
        "family_sequence_centroids": family_model["sequence_centroids"],
        "family_summaries": family_summaries,
        "slot_summaries": slot_summaries,
        "top_atoms": top_atoms,
    }
    return ldi_only, exact_stats, artifact


def _run_dense_eval(ids: torch.Tensor, num_windows: int) -> dict[str, object]:
    model = _load_model()
    _reset_peaks()
    metrics = _evaluate(model, ids, num_windows)
    eval_vram = _vram()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics, "eval_vram": eval_vram, "eval_peak_mb": _peak(eval_vram)}


def _run_preencoded_eval(ids: torch.Tensor, cache_path: Path, strategy: str, num_windows: int) -> dict[str, object]:
    model = _load_model()
    encodings = load_grouped_encoding_map(cache_path)
    _reset_peaks()
    layer_stats = replace_with_preencoded_packed_block_rvq(model, encodings, matmul_strategy=strategy)
    replace_vram = _vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peaks()
    metrics = _evaluate(model, ids, num_windows)
    eval_vram = _vram()
    avg_rel_mse = sum(item["rel_mse"] for item in layer_stats) / max(len(layer_stats), 1)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "replace_vram": replace_vram,
        "eval_vram": eval_vram,
        "replace_peak_mb": _peak(replace_vram),
        "eval_peak_mb": _peak(eval_vram),
        "avg_layer_rel_mse": float(avg_rel_mse),
        "layer_stats": layer_stats,
    }


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
    parser.add_argument("--train-samples", type=int, default=131072)
    parser.add_argument("--family-iters", type=int, default=6)
    parser.add_argument("--atom-iters", type=int, default=6)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--sequence-weight", type=float, default=0.35)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_asm_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ldi_current_l54_gate_up.pt"))
    parser.add_argument("--ldi-only-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ldi_only_l54_gate_up.pt"))
    parser.add_argument("--ldi-artifact", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ldi_isa_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ldi_proto_summary.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)

    ldi_only_encodings: dict[str, GroupedBlockRVQEncoding] = {}
    wal_ldi_rows = []
    ldi_only_compare = []
    artifacts = {}
    for name in TARGETS:
        ldi_only, exact_stats, artifact = _build_ldi_layer(name, current_enc[name], args)
        ldi_only_encodings[name] = ldi_only
        wal_ldi_rows.append(exact_stats)
        ldi_only_compare.append({"name": name, **_compare_encodings(current_enc[name], ldi_only)})
        artifacts[name] = artifact
        print(
            f"[wal-ldi] {name}: avg_program={exact_stats['avg_program_length']:.3f}/{exact_stats['raw_program_length']} "
            f"low_calls={exact_stats['avg_low_level_calls']:.3f} entropy={exact_stats['family_entropy']:.3f} "
            f"ldi_only_match={exact_stats['ldi_only_token_match']:.3f}",
            flush=True,
        )

    ldi_artifact = Path(args.ldi_artifact)
    ldi_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(TARGETS),
            "num_families": int(args.num_families),
            "num_slot_variants": int(args.num_slot_variants),
            "artifacts": artifacts,
        },
        ldi_artifact,
    )
    ldi_only_cache = Path(args.ldi_only_cache)
    save_grouped_encoding_map(ldi_only_cache, ldi_only_encodings)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] ldi_only via {args.matmul_strategy}", flush=True)
    ldi_only_eval = _run_preencoded_eval(ids, ldi_only_cache, args.matmul_strategy, args.num_windows)
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
        "semantic_weight": float(args.semantic_weight),
        "sequence_weight": float(args.sequence_weight),
        "dense": dense,
        "current_cache": str(current_cache),
        "ldi_only_cache": str(ldi_only_cache),
        "ldi_artifact": str(ldi_artifact),
        "wal_ldi": wal_ldi_rows,
        "ldi_only_compare": ldi_only_compare,
        "current_eval": current_eval,
        "exact_eval": exact_eval,
        "ldi_only_eval": ldi_only_eval,
        "delta_exact": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
        },
        "delta_ldi_only": {
            "ppl_delta": float(ldi_only_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(ldi_only_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(ldi_only_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  dense:         ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:       ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_ldi_exact: ppl={exact_eval['metrics']['perplexity']:.4f} tok/s={exact_eval['metrics']['tok_s']:.2f} peak_mb={exact_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  ldi_only:      ppl={ldi_only_eval['metrics']['perplexity']:.4f} tok/s={ldi_only_eval['metrics']['tok_s']:.2f} peak_mb={ldi_only_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row, compare in zip(wal_ldi_rows, ldi_only_compare):
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"low_calls={row['avg_low_level_calls']:.3f} family_entropy={row['family_entropy']:.3f} "
            f"ldi_only_rel_mse={compare['recon_rel_mse']:.6f}",
            flush=True,
        )
    print(
        f"  delta(ldi_only-current): ppl={result['delta_ldi_only']['ppl_delta']:+.6f} "
        f"tok/s={result['delta_ldi_only']['tok_s_delta']:+.2f} peak_mb={result['delta_ldi_only']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()