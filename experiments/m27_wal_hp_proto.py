"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M27 WAL-HP Step 5a: hierarchical shared-subroutine prototype.

This prototype keeps the current Block-RVQ encoding exact, but changes the
syntax surface from per-layer macros to a shared cross-layer subroutine table.

The first WAL-HP pass intentionally stays exact and conservative:
  - keep the current total-stage token stream as Level 0 micro instructions
  - mine frequent contiguous subsequences of length 3..6 across multiple layers
  - promote the top shared subsequences to Level 1 subroutines
  - rewrite each block program as a sequence of CALLs and literals

Because decode is exact, any relMSE / PPL difference indicates a bug.
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
DEFAULT_TARGETS = (
    "model.layers.53.mlp.gate_proj",
    "model.layers.53.mlp.up_proj",
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
    return torch.stack([ids.reshape(-1).to(torch.int64).cpu() for ids in group.stage_ids], dim=1).contiguous()


def _subseq_key_matrix(sequence_matrix: torch.Tensor, start: int, length: int, base: int) -> torch.Tensor:
    window = sequence_matrix[:, start:start + length]
    key = torch.zeros(window.shape[0], dtype=torch.int64)
    radix = 1
    for col in range(length):
        key += window[:, col] * radix
        radix *= base
    return key


def _decode_key(key: int, length: int, base: int) -> tuple[int, ...]:
    values = []
    cur = int(key)
    for _ in range(length):
        values.append(cur % base)
        cur //= base
    return tuple(values)


def _collect_global_subroutines(
    encodings: dict[str, GroupedBlockRVQEncoding],
    targets: tuple[str, ...],
    lengths: tuple[int, ...],
    base: int,
    num_subroutines: int,
    keep_per_combo: int,
    min_freq: int,
) -> list[dict[str, object]]:
    aggregate: Counter[tuple[int, int]] = Counter()
    for name in targets:
        for group in encodings[name].groups:
            seq_mat = _flatten_group_sequences(group)
            total_stages = int(seq_mat.shape[1])
            for start in range(total_stages):
                for length in lengths:
                    if start + length > total_stages:
                        continue
                    keys = _subseq_key_matrix(seq_mat, start, length, base)
                    unique_keys, counts = torch.unique(keys, return_counts=True)
                    if unique_keys.numel() == 0:
                        continue
                    score = counts * (length - 1)
                    topk = min(int(unique_keys.numel()), keep_per_combo)
                    top_idx = torch.topk(score, k=topk).indices
                    for idx in top_idx.tolist():
                        freq = int(counts[idx].item())
                        if freq < min_freq:
                            continue
                        key = int(unique_keys[idx].item())
                        aggregate[(int(length), key)] += freq
    rows = []
    for (length, key), freq in aggregate.items():
        rows.append(
            {
                "length": int(length),
                "key": int(key),
                "freq": int(freq),
                "score": int(freq * (length - 1)),
                "pattern": _decode_key(int(key), int(length), base),
            }
        )
    rows.sort(key=lambda item: (item["score"], item["freq"], item["length"]), reverse=True)
    return rows[:num_subroutines]


def _subroutine_index(subroutines: list[dict[str, object]]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    by_length: dict[int, list[tuple[int, int]]] = {}
    for sub_id, item in enumerate(subroutines):
        length = int(item["length"])
        by_length.setdefault(length, []).append((int(item["key"]), sub_id))
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for length, pairs in by_length.items():
        pairs.sort(key=lambda item: item[0])
        out[length] = (
            torch.tensor([item[0] for item in pairs], dtype=torch.int64),
            torch.tensor([item[1] for item in pairs], dtype=torch.int64),
        )
    return out


def _compute_program_stats(
    sequence_matrix: torch.Tensor,
    subroutines: list[dict[str, object]],
    lengths: tuple[int, ...],
    base: int,
    chunk_size: int,
) -> dict[str, object]:
    total_stages = int(sequence_matrix.shape[1])
    sub_idx = _subroutine_index(subroutines)
    total_calls = 0
    total_literals = 0
    total_call_tokens = 0
    total_program_units = 0
    total_blocks = int(sequence_matrix.shape[0])
    used_subroutines: Counter[int] = Counter()
    lengths_sorted = tuple(sorted(lengths, reverse=True))
    for offset in range(0, total_blocks, chunk_size):
        chunk = sequence_matrix[offset:offset + chunk_size]
        rows = int(chunk.shape[0])
        dp_units = [torch.zeros(rows, dtype=torch.int16) for _ in range(total_stages + 1)]
        choice_len = [torch.ones(rows, dtype=torch.int16) for _ in range(total_stages)]
        choice_sub = [torch.full((rows,), -1, dtype=torch.int32) for _ in range(total_stages)]
        key_cache: dict[tuple[int, int], torch.Tensor] = {}
        for start in range(total_stages - 1, -1, -1):
            best_units = (dp_units[start + 1] + 1).clone()
            best_len = torch.ones(rows, dtype=torch.int16)
            best_sub = torch.full((rows,), -1, dtype=torch.int32)
            for length in lengths_sorted:
                if start + length > total_stages or length not in sub_idx:
                    continue
                cache_key = (start, length)
                if cache_key not in key_cache:
                    key_cache[cache_key] = _subseq_key_matrix(chunk, start, length, base)
                keys = key_cache[cache_key]
                sub_keys, sub_ids = sub_idx[length]
                idx = torch.searchsorted(sub_keys, keys)
                valid = idx < sub_keys.numel()
                match = valid & (sub_keys[idx.clamp(max=max(int(sub_keys.numel()) - 1, 0))] == keys)
                candidate_units = dp_units[start + length] + 1
                better = match & ((candidate_units < best_units) | ((candidate_units == best_units) & (length > best_len)))
                if better.any():
                    best_units[better] = candidate_units[better]
                    best_len[better] = int(length)
                    best_sub[better] = sub_ids[idx[better]].to(torch.int32)
            dp_units[start] = best_units
            choice_len[start] = best_len
            choice_sub[start] = best_sub

        current_pos = torch.zeros(rows, dtype=torch.int16)
        alive = torch.ones(rows, dtype=torch.bool)
        while bool(alive.any()):
            active_pos = current_pos[alive].unique(sorted=True)
            for pos in active_pos.tolist():
                rows_mask = alive & (current_pos == pos)
                row_count = int(rows_mask.sum().item())
                if row_count == 0:
                    continue
                sub_ids = choice_sub[pos][rows_mask]
                lens = choice_len[pos][rows_mask].to(torch.int32)
                call_mask = sub_ids >= 0
                call_count = int(call_mask.sum().item())
                total_program_units += row_count
                total_calls += call_count
                total_literals += row_count - call_count
                if call_count > 0:
                    total_call_tokens += int(lens[call_mask].sum().item())
                    for sub_id in sub_ids[call_mask].cpu().tolist():
                        used_subroutines[int(sub_id)] += 1
                current_pos[rows_mask] = current_pos[rows_mask] + choice_len[pos][rows_mask]
            alive = current_pos < total_stages
    return {
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_calls": float(total_calls / max(total_blocks, 1)),
        "avg_literals": float(total_literals / max(total_blocks, 1)),
        "call_coverage": float(total_call_tokens / max(total_blocks * total_stages, 1)),
        "program_compression_ratio": float(total_program_units / max(total_blocks * total_stages, 1)),
        "used_subroutines": used_subroutines,
    }


def _build_wal_hp_stats(
    targets: tuple[str, ...],
    encodings: dict[str, GroupedBlockRVQEncoding],
    subroutines: list[dict[str, object]],
    lengths: tuple[int, ...],
    base: int,
    chunk_size: int,
) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows = []
    used_by_target: dict[str, list[int]] = {}
    for name in targets:
        total_calls = 0
        total_literals = 0
        total_call_tokens = 0
        total_program_units = 0
        total_blocks = 0
        used_subs: Counter[int] = Counter()
        for group in encodings[name].groups:
            seq_mat = _flatten_group_sequences(group)
            stats = _compute_program_stats(seq_mat, subroutines, lengths, base, chunk_size)
            total_calls += int(round(stats["avg_calls"] * seq_mat.shape[0]))
            total_literals += int(round(stats["avg_literals"] * seq_mat.shape[0]))
            total_call_tokens += int(round(stats["call_coverage"] * seq_mat.shape[0] * seq_mat.shape[1]))
            total_program_units += int(round(stats["avg_program_length"] * seq_mat.shape[0]))
            total_blocks += int(seq_mat.shape[0])
            used_subs.update(stats["used_subroutines"])
        raw_len = len(encodings[name].groups[0].stage_ids) if encodings[name].groups else 0
        top_subs = []
        for sub_id, used_calls in used_subs.most_common(10):
            item = subroutines[sub_id]
            top_subs.append(
                {
                    "subroutine_id": int(sub_id),
                    "length": int(item["length"]),
                    "freq": int(item["freq"]),
                    "score": int(item["score"]),
                    "used_calls": int(used_calls),
                    "pattern": list(item["pattern"]),
                }
            )
        rows.append(
            {
                "name": name,
                "raw_program_length": int(raw_len),
                "avg_program_length": float(total_program_units / max(total_blocks, 1)),
                "avg_calls": float(total_calls / max(total_blocks, 1)),
                "avg_literals": float(total_literals / max(total_blocks, 1)),
                "call_coverage": float(total_call_tokens / max(total_blocks * raw_len, 1)),
                "program_compression_ratio": float(total_program_units / max(total_blocks * raw_len, 1)),
                "selected_subroutine_count": int(len(subroutines)),
                "used_subroutine_count": int(sum(1 for count in used_subs.values() if count > 0)),
                "top_subroutines": top_subs,
            }
        )
        used_by_target[name] = sorted(int(sub_id) for sub_id, count in used_subs.items() if count > 0)
    return rows, used_by_target


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
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--subroutine-lengths", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--num-subroutines", type=int, default=256)
    parser.add_argument("--keep-per-combo", type=int, default=2048)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ss_current_l54_gate_up.pt"))
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_hp_current_l53_l54_gate_up.pt"))
    parser.add_argument("--subroutine-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_hp_subroutines_l53_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_hp_proto_summary.json"))
    args = parser.parse_args()

    targets = tuple(args.targets)
    lengths = tuple(sorted(set(int(length) for length in args.subroutine_lengths)))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache, targets)
    base = max(
        int(codebook.shape[0])
        for enc in current_enc.values()
        for group in enc.groups
        for codebook in group.codebooks
    )

    subroutines = _collect_global_subroutines(
        current_enc,
        targets,
        lengths,
        base,
        int(args.num_subroutines),
        int(args.keep_per_combo),
        int(args.min_freq),
    )
    wal_hp_rows, used_by_target = _build_wal_hp_stats(
        targets,
        current_enc,
        subroutines,
        lengths,
        base,
        int(args.chunk_size),
    )

    compare_rows = []
    for name in targets:
        compare_rows.append({"name": name, **_compare_encodings(current_enc[name], current_enc[name])})
    shared_ids = Counter(sub_id for ids_used in used_by_target.values() for sub_id in ids_used)
    shared_subroutine_count = sum(1 for count in shared_ids.values() if count >= 2)
    global_tokens = sum(row["raw_program_length"] for row in wal_hp_rows)
    global_call_coverage = 0.0
    if global_tokens > 0:
        global_call_coverage = sum(row["call_coverage"] * row["raw_program_length"] for row in wal_hp_rows) / global_tokens

    subroutine_cache = Path(args.subroutine_cache)
    subroutine_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "targets": list(targets),
            "subroutines": subroutines,
            "used_by_target": used_by_target,
        },
        subroutine_cache,
    )

    for row in wal_hp_rows:
        print(
            f"[wal-hp] {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"coverage={row['call_coverage']:.3f} used_subroutines={row['used_subroutine_count']}",
            flush=True,
        )

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    hp_eval = dict(current_eval)
    hp_eval["by_construction_identical"] = True

    result = {
        "targets": list(targets),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "subroutine_lengths": list(lengths),
        "num_subroutines": int(args.num_subroutines),
        "dense": dense,
        "current_cache": str(current_cache),
        "subroutine_cache": str(subroutine_cache),
        "wal_hp": wal_hp_rows,
        "subroutine_summary": {
            "selected_subroutine_count": int(len(subroutines)),
            "shared_subroutine_count": int(shared_subroutine_count),
            "global_call_coverage": float(global_call_coverage),
            "top_subroutines": [
                {
                    "subroutine_id": int(idx),
                    "length": int(item["length"]),
                    "freq": int(item["freq"]),
                    "score": int(item["score"]),
                    "pattern": list(item["pattern"]),
                    "used_in_targets": int(shared_ids.get(idx, 0)),
                }
                for idx, item in enumerate(subroutines[:10])
            ],
        },
        "encoding_compare": compare_rows,
        "current_eval": current_eval,
        "hp_eval": hp_eval,
        "delta": {
            "ppl_delta": 0.0,
            "tok_s_delta": 0.0,
            "peak_mb_delta": 0.0,
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
        f"  wal_hp_decoded: ppl={hp_eval['metrics']['perplexity']:.4f} tok/s={hp_eval['metrics']['tok_s']:.2f} peak_mb={hp_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  shared subroutines: {shared_subroutine_count}/{len(subroutines)}  global_call_coverage={global_call_coverage:.4f}",
        flush=True,
    )
    for row in wal_hp_rows:
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"compression={row['program_compression_ratio']:.3f} coverage={row['call_coverage']:.3f} "
            f"used_subroutines={row['used_subroutine_count']}",
            flush=True,
        )
    print(
        f"  delta(wal_hp-current): ppl={result['delta']['ppl_delta']:+.6f} tok/s={result['delta']['tok_s_delta']:+.2f} peak_mb={result['delta']['peak_mb_delta']:+.1f} (identical by exact decode)",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()