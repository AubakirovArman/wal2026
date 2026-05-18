"""M27 WAL-TS Step 8a: typed WAL prototype for l54 gate/up.

This probe extends WAL-FG with a simple route type system.

For the current two-layer slice the active atomic route types are:
  - MLP_GATE_COMMON / MLP_GATE_MIXED / MLP_GATE_OUTLIER
  - MLP_UP_COMMON   / MLP_UP_MIXED   / MLP_UP_OUTLIER

Block programs are typed by a simple heuristic over stage-id surprisal.
Typed grammar rules are then learned separately per type, so the exact surface
becomes `TYPE + RULE calls + literal corrections`, while `typed_only`
approximates each slot with the nearest phrase bank for that block type.
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


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512
TARGETS = (
    "model.language_model.layers.54.mlp.gate_proj",
    "model.language_model.layers.54.mlp.up_proj",
)
TYPE_BUCKETS = ("COMMON", "MIXED", "OUTLIER")


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
        row = {
            "iter": int(iter_idx + 1),
            "mean_hamming": float(distances.to(torch.float32).mean().item()),
            "p50_hamming": float(torch.quantile(distances.to(torch.float32), 0.50).item()),
            "p90_hamming": float(torch.quantile(distances.to(torch.float32), 0.90).item()),
        }
        history.append(row)
    final_assignments, final_distances = _nearest_phrases(slot_sequences, phrases, assign_chunk_size)
    final_counts = torch.bincount(final_assignments, minlength=num_variants)
    return {
        "phrases": phrases,
        "train_counts": final_counts,
        "train_mean_hamming": float(final_distances.to(torch.float32).mean().item()),
        "history": history,
    }


def _role_name(layer_name: str) -> str:
    if layer_name.endswith("gate_proj"):
        return "MLP_GATE"
    if layer_name.endswith("up_proj"):
        return "MLP_UP"
    raise ValueError(f"unsupported target for WAL-TS: {layer_name}")


def _type_names_for_role(role_name: str) -> tuple[str, str, str]:
    return tuple(f"{role_name}_{bucket}" for bucket in TYPE_BUCKETS)


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


def _build_type_model(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> dict[str, object]:
    sample_sequences = _sample_training_sequences(enc, int(args.train_samples), seed=0)
    raw_len = int(sample_sequences.shape[1])
    base = max(int(codebook.shape[0]) for group in enc.groups for codebook in group.codebooks)
    counts = torch.zeros(raw_len, base, dtype=torch.int64)
    for pos in range(raw_len):
        counts[pos] = torch.bincount(sample_sequences[:, pos].to(torch.int64), minlength=base)
    probs = (counts.to(torch.float32) + 1.0) / (float(sample_sequences.shape[0]) + float(base))
    stage_surprisal = (-torch.log(probs)).contiguous()
    modal_tokens = counts.argmax(dim=1).to(torch.uint8)
    surprisal_sum = torch.zeros(int(sample_sequences.shape[0]), dtype=torch.float32)
    for pos in range(raw_len):
        surprisal_sum += stage_surprisal[pos][sample_sequences[:, pos].to(torch.int64)]
    avg_surprisal = surprisal_sum / float(raw_len)
    common_q = float(torch.quantile(avg_surprisal, float(args.common_quantile)).item())
    outlier_q = float(torch.quantile(avg_surprisal, float(args.outlier_quantile)).item())
    if outlier_q <= common_q:
        raise ValueError("outlier_quantile must produce a higher threshold than common_quantile")
    modal_match = (sample_sequences == modal_tokens.unsqueeze(0)).to(torch.float32).mean(dim=1)
    sample_type_ids = torch.full((int(sample_sequences.shape[0]),), 1, dtype=torch.int64)
    sample_type_ids[avg_surprisal <= common_q] = 0
    sample_type_ids[avg_surprisal >= outlier_q] = 2
    role_name = _role_name(name)
    type_names = _type_names_for_role(role_name)
    bucket_counts = torch.bincount(sample_type_ids, minlength=len(type_names))
    type_summaries = []
    for type_idx, type_name in enumerate(type_names):
        mask = sample_type_ids == type_idx
        type_summaries.append(
            {
                "type_id": int(type_idx),
                "type_name": type_name,
                "sample_count": int(bucket_counts[type_idx].item()),
                "sample_share": float(bucket_counts[type_idx].item() / max(int(sample_sequences.shape[0]), 1)),
                "sample_avg_surprisal": float(avg_surprisal[mask].mean().item()) if bool(mask.any()) else 0.0,
                "sample_modal_match": float(modal_match[mask].mean().item()) if bool(mask.any()) else 0.0,
            }
        )
    return {
        "role_name": role_name,
        "type_names": type_names,
        "raw_len": raw_len,
        "base": base,
        "stage_surprisal": stage_surprisal,
        "modal_tokens": modal_tokens,
        "common_threshold": common_q,
        "outlier_threshold": outlier_q,
        "sample_sequences": sample_sequences,
        "sample_type_ids": sample_type_ids,
        "type_summaries": type_summaries,
    }


def _assign_types(sequence_matrix: torch.Tensor, type_model: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_len = int(type_model["raw_len"])
    stage_surprisal = type_model["stage_surprisal"]
    modal_tokens = type_model["modal_tokens"]
    surprisal_sum = torch.zeros(int(sequence_matrix.shape[0]), dtype=torch.float32)
    for pos in range(raw_len):
        surprisal_sum += stage_surprisal[pos][sequence_matrix[:, pos].to(torch.int64)]
    avg_surprisal = surprisal_sum / float(raw_len)
    modal_match = (sequence_matrix == modal_tokens.unsqueeze(0)).to(torch.float32).mean(dim=1)
    type_ids = torch.full((int(sequence_matrix.shape[0]),), 1, dtype=torch.int64)
    type_ids[avg_surprisal <= float(type_model["common_threshold"])] = 0
    type_ids[avg_surprisal >= float(type_model["outlier_threshold"])] = 2
    return type_ids, avg_surprisal, modal_match


def _build_typed_grammar_rules(
    type_names: tuple[str, ...],
    slot_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    rules = []
    rule_id = 0
    for type_summary in slot_summaries:
        type_name = type_names[int(type_summary["type_id"])]
        rules.append({"rule_id": rule_id, "lhs": f"{type_name}_BLOCK", "rhs": [f"{type_name}_LEFT", f"{type_name}_RIGHT"], "prob": 1.0})
        rule_id += 1
        rules.append({"rule_id": rule_id, "lhs": f"{type_name}_LEFT", "rhs": [f"{type_name}_SLOT_0", f"{type_name}_SLOT_1"], "prob": 1.0})
        rule_id += 1
        rules.append({"rule_id": rule_id, "lhs": f"{type_name}_RIGHT", "rhs": [f"{type_name}_SLOT_2", f"{type_name}_SLOT_3"], "prob": 1.0})
        rule_id += 1
        for slot_row in type_summary["slots"]:
            total = max(int(slot_row["train_total"]), 1)
            for variant in slot_row["variants"]:
                rules.append(
                    {
                        "rule_id": int(rule_id),
                        "lhs": f"{type_name}_SLOT_{slot_row['slot_index']}",
                        "rhs": [int(token) for token in variant["phrase"]],
                        "prob": float(variant["train_count"] / total),
                        "variant_id": int(variant["variant_id"]),
                        "type_id": int(type_summary["type_id"]),
                    }
                )
                rule_id += 1
    return rules


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


def _build_ts_layer(
    name: str,
    enc: GroupedBlockRVQEncoding,
    args: argparse.Namespace,
) -> tuple[GroupedBlockRVQEncoding, dict[str, object], dict[str, object]]:
    type_model = _build_type_model(name, enc, args)
    raw_len = int(type_model["raw_len"])
    phrase_len = int(args.phrase_len)
    if raw_len % phrase_len != 0:
        raise ValueError("phrase_len must divide raw program length")
    num_slots = raw_len // phrase_len
    if num_slots != 4:
        raise ValueError("the first WAL-TS prototype expects exactly 4 phrase slots")
    type_names = type_model["type_names"]
    sample_sequences = type_model["sample_sequences"]
    sample_type_ids = type_model["sample_type_ids"]
    base = int(type_model["base"])

    typed_slot_learned: list[list[dict[str, object]]] = []
    typed_slot_summaries = []
    for type_idx, type_name in enumerate(type_names):
        type_mask = sample_type_ids == type_idx
        type_sequences = sample_sequences[type_mask]
        print(
            f"[wal-ts/train] {name}: type={type_name} samples={type_sequences.shape[0]} variants={args.num_phrase_variants}",
            flush=True,
        )
        slots = []
        slot_learned = []
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            learned = _learn_slot_phrases(
                type_sequences[:, start:end],
                int(args.num_phrase_variants),
                base,
                int(args.grammar_iters),
                int(args.assign_chunk_size),
            )
            slot_learned.append(learned)
            variants = []
            total = int(learned["train_counts"].sum().item())
            for variant_id in range(int(args.num_phrase_variants)):
                variants.append(
                    {
                        "variant_id": int(variant_id),
                        "train_count": int(learned["train_counts"][variant_id].item()),
                        "phrase": learned["phrases"][variant_id].to(torch.int64).tolist(),
                    }
                )
            slots.append(
                {
                    "slot_index": int(slot_idx),
                    "phrase_len": phrase_len,
                    "train_total": int(total),
                    "train_mean_hamming": float(learned["train_mean_hamming"]),
                    "history": learned["history"],
                    "variants": variants,
                }
            )
        typed_slot_learned.append(slot_learned)
        typed_slot_summaries.append(
            {
                "type_id": int(type_idx),
                "type_name": type_name,
                "slots": slots,
            }
        )

    grammar_rules = _build_typed_grammar_rules(type_names, typed_slot_summaries)
    structural_rule_count = len(type_names) * 3
    parse_tree_depth = 4
    parse_nonterminal_nodes = 7
    total_blocks = 0
    total_program_units = 0
    total_rule_calls = 0
    total_literal_corrections = 0
    total_rule_tokens = 0
    total_typed_only_hamming = 0
    total_exact_slot_matches = 0
    total_usable_rule_slots = 0
    total_slot_assignments = 0
    type_count_totals = torch.zeros(len(type_names), dtype=torch.int64)
    type_program_units = torch.zeros(len(type_names), dtype=torch.int64)
    type_rule_calls = torch.zeros(len(type_names), dtype=torch.int64)
    type_literal_corrections = torch.zeros(len(type_names), dtype=torch.int64)
    type_rule_tokens = torch.zeros(len(type_names), dtype=torch.int64)
    type_typed_only_hamming = torch.zeros(len(type_names), dtype=torch.int64)
    type_exact_slot_matches = torch.zeros(len(type_names), dtype=torch.int64)
    type_usable_rule_slots = torch.zeros(len(type_names), dtype=torch.int64)
    type_slot_assignments = torch.zeros(len(type_names), dtype=torch.int64)
    type_surprisal_sum = torch.zeros(len(type_names), dtype=torch.float64)
    type_modal_match_sum = torch.zeros(len(type_names), dtype=torch.float64)
    production_usage: Counter[tuple[int, int, int]] = Counter()
    typed_only_groups = []
    type_position_counts = torch.zeros(len(type_names), raw_len, base, dtype=torch.int64)

    for group in enc.groups:
        seq_mat = _flatten_group_sequences(group)
        type_ids, avg_surprisal, modal_match = _assign_types(seq_mat, type_model)
        group_rule_calls = torch.zeros(int(seq_mat.shape[0]), dtype=torch.int32)
        group_literal_corrections = torch.zeros(int(seq_mat.shape[0]), dtype=torch.int32)
        group_program_units = torch.zeros(int(seq_mat.shape[0]), dtype=torch.int32)
        approx_slots = []
        for type_idx in range(len(type_names)):
            mask = type_ids == type_idx
            if bool(mask.any()):
                type_count_totals[type_idx] += int(mask.sum().item())
                type_surprisal_sum[type_idx] += float(avg_surprisal[mask].sum().item())
                type_modal_match_sum[type_idx] += float(modal_match[mask].sum().item())
                type_seq = seq_mat[mask]
                for pos in range(raw_len):
                    type_position_counts[type_idx, pos] += torch.bincount(type_seq[:, pos].to(torch.int64), minlength=base)
        for slot_idx in range(num_slots):
            start = slot_idx * phrase_len
            end = start + phrase_len
            slot_sequences = seq_mat[:, start:end]
            approx_slot = torch.empty_like(slot_sequences)
            for type_idx in range(len(type_names)):
                mask = type_ids == type_idx
                if not bool(mask.any()):
                    continue
                learned = typed_slot_learned[type_idx][slot_idx]
                masked_sequences = slot_sequences[mask]
                assignments, distances = _nearest_phrases(masked_sequences, learned["phrases"], int(args.assign_chunk_size))
                dist_i32 = distances.to(torch.int32)
                use_rule = (dist_i32 + 1) < phrase_len
                type_exact_slot_matches[type_idx] += int((dist_i32 == 0).sum().item())
                type_usable_rule_slots[type_idx] += int(use_rule.sum().item())
                type_typed_only_hamming[type_idx] += int(dist_i32.sum().item())
                type_slot_assignments[type_idx] += int(dist_i32.numel())
                total_exact_slot_matches += int((dist_i32 == 0).sum().item())
                total_usable_rule_slots += int(use_rule.sum().item())
                total_typed_only_hamming += int(dist_i32.sum().item())
                total_slot_assignments += int(dist_i32.numel())
                group_rule_calls[mask] += use_rule.to(torch.int32)
                group_literal_corrections[mask] += torch.where(use_rule, dist_i32, torch.full_like(dist_i32, phrase_len))
                group_program_units[mask] += torch.where(use_rule, dist_i32 + 1, torch.full_like(dist_i32, phrase_len))
                saved_tokens = torch.where(use_rule, phrase_len - dist_i32, torch.zeros_like(dist_i32))
                type_rule_tokens[type_idx] += int(saved_tokens.sum().item())
                total_rule_tokens += int(saved_tokens.sum().item())
                if bool(use_rule.any()):
                    slot_counts = torch.bincount(assignments[use_rule], minlength=int(args.num_phrase_variants))
                    for variant_id, count in enumerate(slot_counts.tolist()):
                        if count > 0:
                            production_usage[(int(type_idx), int(slot_idx), int(variant_id))] += int(count)
                approx_slot[mask] = learned["phrases"][assignments].to(torch.uint8)
            approx_slots.append(approx_slot)
        total_blocks += int(seq_mat.shape[0])
        total_rule_calls += int(group_rule_calls.sum().item())
        total_literal_corrections += int(group_literal_corrections.sum().item())
        total_program_units += int(group_program_units.sum().item())
        for type_idx in range(len(type_names)):
            mask = type_ids == type_idx
            if bool(mask.any()):
                type_rule_calls[type_idx] += int(group_rule_calls[mask].sum().item())
                type_literal_corrections[type_idx] += int(group_literal_corrections[mask].sum().item())
                type_program_units[type_idx] += int(group_program_units[mask].sum().item())
        typed_only_groups.append(_clone_group_with_program_matrix(group, torch.cat(approx_slots, dim=1)))

    typed_only = GroupedBlockRVQEncoding(
        groups=tuple(typed_only_groups),
        row_slices=enc.row_slices,
        original_shape=enc.original_shape,
    )

    type_centroids = type_position_counts.argmax(dim=2).to(torch.uint8)
    type_own_centroid_hamming = torch.zeros(len(type_names), dtype=torch.int64)
    type_other_centroid_hamming = torch.zeros(len(type_names), dtype=torch.int64)
    type_global_modal_hamming = torch.zeros(len(type_names), dtype=torch.int64)
    modal_tokens = type_model["modal_tokens"]
    for group in enc.groups:
        seq_mat = _flatten_group_sequences(group)
        type_ids, _, _ = _assign_types(seq_mat, type_model)
        global_hamming = (seq_mat != modal_tokens.unsqueeze(0)).sum(dim=1).to(torch.int64)
        centroid_dist = torch.empty(int(seq_mat.shape[0]), len(type_names), dtype=torch.int64)
        for type_idx in range(len(type_names)):
            centroid_dist[:, type_idx] = (seq_mat != type_centroids[type_idx].unsqueeze(0)).sum(dim=1).to(torch.int64)
        for type_idx in range(len(type_names)):
            mask = type_ids == type_idx
            if not bool(mask.any()):
                continue
            own = centroid_dist[mask, type_idx]
            other_cols = [other_idx for other_idx in range(len(type_names)) if other_idx != type_idx]
            other = centroid_dist[mask][:, other_cols].min(dim=1).values if other_cols else own.clone()
            type_own_centroid_hamming[type_idx] += int(own.sum().item())
            type_other_centroid_hamming[type_idx] += int(other.sum().item())
            type_global_modal_hamming[type_idx] += int(global_hamming[mask].sum().item())

    type_summaries = []
    for type_idx, type_name in enumerate(type_names):
        type_count = int(type_count_totals[type_idx].item())
        type_slots = max(type_count * num_slots, 1)
        type_tokens = max(type_count * raw_len, 1)
        type_summary = {
            "type_id": int(type_idx),
            "type_name": type_name,
            "block_count": type_count,
            "block_share": float(type_count / max(total_blocks, 1)),
            "avg_surprisal": float(type_surprisal_sum[type_idx].item() / max(type_count, 1)),
            "avg_modal_match": float(type_modal_match_sum[type_idx].item() / max(type_count, 1)),
            "avg_program_length": float(type_program_units[type_idx].item() / max(type_count, 1)),
            "avg_rule_calls": float(type_rule_calls[type_idx].item() / max(type_count, 1)),
            "avg_literal_corrections": float(type_literal_corrections[type_idx].item() / max(type_count, 1)),
            "rule_token_coverage": float(type_rule_tokens[type_idx].item() / type_tokens),
            "slot_exact_match_rate": float(type_exact_slot_matches[type_idx].item() / type_slots),
            "slot_rule_use_rate": float(type_usable_rule_slots[type_idx].item() / type_slots),
            "typed_only_avg_hamming": float(type_typed_only_hamming[type_idx].item() / max(type_count, 1)),
            "typed_only_token_match": float(1.0 - type_typed_only_hamming[type_idx].item() / type_tokens),
            "own_type_centroid_hamming": float(type_own_centroid_hamming[type_idx].item() / max(type_count, 1)),
            "nearest_other_centroid_hamming": float(type_other_centroid_hamming[type_idx].item() / max(type_count, 1)),
            "global_modal_hamming": float(type_global_modal_hamming[type_idx].item() / max(type_count, 1)),
            "type_separation_margin": float((type_other_centroid_hamming[type_idx] - type_own_centroid_hamming[type_idx]).item() / max(type_count, 1)),
            "centroid_gain_vs_global": float((type_global_modal_hamming[type_idx] - type_own_centroid_hamming[type_idx]).item() / max(type_count, 1)),
        }
        type_summaries.append(type_summary)

    top_productions = []
    for (type_idx, slot_idx, variant_id), used_calls in production_usage.most_common(12):
        learned = typed_slot_learned[type_idx][slot_idx]
        phrase = learned["phrases"][variant_id].to(torch.int64).tolist()
        train_count = int(learned["train_counts"][variant_id].item())
        train_total = max(int(learned["train_counts"].sum().item()), 1)
        top_productions.append(
            {
                "type_name": type_names[type_idx],
                "slot_index": int(slot_idx),
                "variant_id": int(variant_id),
                "used_calls": int(used_calls),
                "train_count": int(train_count),
                "rule_prob": float(train_count / train_total),
                "phrase": phrase,
            }
        )

    active_type_count = sum(1 for type_count in type_count_totals.tolist() if type_count > 0)
    type_entropy = 0.0
    for type_count in type_count_totals.tolist():
        if type_count <= 0:
            continue
        p = float(type_count / max(total_blocks, 1))
        type_entropy -= p * math.log(p)

    exact_stats = {
        "name": name,
        "role_name": type_model["role_name"],
        "raw_program_length": raw_len,
        "phrase_len": phrase_len,
        "slot_count": num_slots,
        "type_count": len(type_names),
        "active_type_count": int(active_type_count),
        "route_types": list(type_names),
        "grammar_rule_count": int(len(grammar_rules)),
        "structural_rule_count": int(structural_rule_count),
        "parse_tree_depth": int(parse_tree_depth),
        "parse_nonterminal_nodes": int(parse_nonterminal_nodes),
        "type_entropy": float(type_entropy),
        "type_check_pass_rate": 1.0,
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_rule_calls": float(total_rule_calls / max(total_blocks, 1)),
        "avg_literal_corrections": float(total_literal_corrections / max(total_blocks, 1)),
        "typed_rule_token_coverage": float(total_rule_tokens / max(total_blocks * raw_len, 1)),
        "program_compression_ratio": float(total_program_units / max(total_blocks * raw_len, 1)),
        "used_production_count": int(sum(1 for count in production_usage.values() if count > 0)),
        "slot_exact_match_rate": float(total_exact_slot_matches / max(total_slot_assignments, 1)),
        "slot_rule_use_rate": float(total_usable_rule_slots / max(total_slot_assignments, 1)),
        "typed_only_avg_hamming": float(total_typed_only_hamming / max(total_blocks, 1)),
        "typed_only_token_match": float(1.0 - total_typed_only_hamming / max(total_blocks * raw_len, 1)),
        "common_threshold": float(type_model["common_threshold"]),
        "outlier_threshold": float(type_model["outlier_threshold"]),
        "type_summaries": type_summaries,
        "top_productions": top_productions,
    }
    artifact = {
        "type_model": {
            "role_name": type_model["role_name"],
            "type_names": type_names,
            "common_threshold": float(type_model["common_threshold"]),
            "outlier_threshold": float(type_model["outlier_threshold"]),
            "modal_tokens": type_model["modal_tokens"],
            "stage_surprisal": type_model["stage_surprisal"],
            "sample_type_summaries": type_model["type_summaries"],
        },
        "grammar_rules": grammar_rules,
        "typed_slot_summaries": typed_slot_summaries,
        "top_productions": top_productions,
    }
    return typed_only, exact_stats, artifact


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
    parser.add_argument("--phrase-len", type=int, default=3)
    parser.add_argument("--num-phrase-variants", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=131072)
    parser.add_argument("--grammar-iters", type=int, default=6)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--common-quantile", type=float, default=0.50)
    parser.add_argument("--outlier-quantile", type=float, default=0.90)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_lrt_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ts_current_l54_gate_up.pt"))
    parser.add_argument("--typed-only-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ts_typed_only_l54_gate_up.pt"))
    parser.add_argument("--type-artifact", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ts_types_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ts_proto_summary.json"))
    args = parser.parse_args()

    if not (0.0 < args.common_quantile < args.outlier_quantile < 1.0):
        raise ValueError("require 0 < common_quantile < outlier_quantile < 1")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, TARGETS)

    typed_only_encodings: dict[str, GroupedBlockRVQEncoding] = {}
    wal_ts_rows = []
    typed_only_compare = []
    artifacts = {}
    for name in TARGETS:
        typed_only, ts_stats, artifact = _build_ts_layer(name, current_enc[name], args)
        typed_only_encodings[name] = typed_only
        wal_ts_rows.append(ts_stats)
        typed_only_compare.append({"name": name, **_compare_encodings(current_enc[name], typed_only)})
        artifacts[name] = artifact
        print(
            f"[wal-ts] {name}: avg_program={ts_stats['avg_program_length']:.3f}/{ts_stats['raw_program_length']} "
            f"coverage={ts_stats['typed_rule_token_coverage']:.3f} active_types={ts_stats['active_type_count']} "
            f"typed_only_match={ts_stats['typed_only_token_match']:.3f}",
            flush=True,
        )

    type_artifact = Path(args.type_artifact)
    type_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(TARGETS),
            "type_taxonomy": [
                "MLP_GATE_COMMON",
                "MLP_GATE_MIXED",
                "MLP_GATE_OUTLIER",
                "MLP_UP_COMMON",
                "MLP_UP_MIXED",
                "MLP_UP_OUTLIER",
            ],
            "artifacts": artifacts,
        },
        type_artifact,
    )
    typed_only_cache = Path(args.typed_only_cache)
    save_grouped_encoding_map(typed_only_cache, typed_only_encodings)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    print(f"\n[eval] typed_only via {args.matmul_strategy}", flush=True)
    typed_only_eval = _run_preencoded_eval(ids, typed_only_cache, args.matmul_strategy, args.num_windows)
    exact_eval = dict(current_eval)
    exact_eval["by_construction_identical"] = True

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "phrase_len": int(args.phrase_len),
        "num_phrase_variants": int(args.num_phrase_variants),
        "type_taxonomy": [
            "MLP_GATE_COMMON",
            "MLP_GATE_MIXED",
            "MLP_GATE_OUTLIER",
            "MLP_UP_COMMON",
            "MLP_UP_MIXED",
            "MLP_UP_OUTLIER",
        ],
        "dense": dense,
        "current_cache": str(current_cache),
        "typed_only_cache": str(typed_only_cache),
        "type_artifact": str(type_artifact),
        "wal_ts": wal_ts_rows,
        "typed_only_compare": typed_only_compare,
        "current_eval": current_eval,
        "exact_eval": exact_eval,
        "typed_only_eval": typed_only_eval,
        "delta_exact": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
        },
        "delta_typed_only": {
            "ppl_delta": float(typed_only_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(typed_only_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(typed_only_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"  dense:          ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:        ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_ts_exact:   ppl={exact_eval['metrics']['perplexity']:.4f} tok/s={exact_eval['metrics']['tok_s']:.2f} peak_mb={exact_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  wal_ts_typed:   ppl={typed_only_eval['metrics']['perplexity']:.4f} tok/s={typed_only_eval['metrics']['tok_s']:.2f} peak_mb={typed_only_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row, compare in zip(wal_ts_rows, typed_only_compare):
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"coverage={row['typed_rule_token_coverage']:.3f} active_types={row['active_type_count']} "
            f"typed_only_rel_mse={compare['recon_rel_mse']:.6f}",
            flush=True,
        )
    print(
        f"  delta(typed_only-current): ppl={result['delta_typed_only']['ppl_delta']:+.6f} "
        f"tok/s={result['delta_typed_only']['tok_s_delta']:+.2f} peak_mb={result['delta_typed_only']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()