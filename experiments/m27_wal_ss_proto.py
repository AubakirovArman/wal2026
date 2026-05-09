"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M27 WAL-SS Step 4a: structured-syntax macro prototype for l54.gate/up.

This is a first-level WAL syntax prototype, not a new runtime.

We keep the current 12x256 Block-RVQ encoding for:
  - model.layers.54.mlp.gate_proj
  - model.layers.54.mlp.up_proj

Then we build a lossless macro layer over the stage-id programs:
  - mine frequent contiguous subsequences of total-stage IDs, length 3..5
  - select the top macro vocabulary by compression score
  - re-encode each block program as `macro calls + literal args`
  - decode back to the exact original stage-id stream

The decisive checks are:
  1. decode must be exact (otherwise quality can change)
  2. average program length must drop enough to justify the syntax layer
  3. the macro vocabulary actually used must be meaningfully smaller than the
     raw per-block token surface

Because the prototype is lossless, any relMSE / PPL change would indicate a bug.
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


def _build_current_cache(args: argparse.Namespace, cache_path: Path) -> dict[str, GroupedBlockRVQEncoding]:
    if cache_path.exists() and not args.rebuild_cache:
        print(f"[cache] load {cache_path}", flush=True)
        return load_grouped_encoding_map(cache_path)
    print(f"[cache] build {cache_path}", flush=True)
    model = _load_model()
    module_map = dict(model.named_modules())
    encodings: dict[str, GroupedBlockRVQEncoding] = {}
    for name in TARGETS:
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


def _mine_macros(sequence_matrix: torch.Tensor, lengths: tuple[int, ...], num_macros: int, base: int) -> list[dict[str, object]]:
    total_stages = int(sequence_matrix.shape[1])
    candidates: list[dict[str, object]] = []
    keep_per_combo = max(int(num_macros) * 2, 64)
    for start in range(total_stages):
        for length in lengths:
            if start + length > total_stages:
                continue
            keys = _subseq_key_matrix(sequence_matrix, start, length, base)
            unique_keys, counts = torch.unique(keys, return_counts=True)
            if unique_keys.numel() == 0:
                continue
            scores = counts * (length - 1)
            topk = min(keep_per_combo, int(unique_keys.numel()))
            top_idx = torch.topk(scores, k=topk).indices
            for idx in top_idx.tolist():
                freq = int(counts[idx].item())
                if freq < 2:
                    continue
                key = int(unique_keys[idx].item())
                candidates.append(
                    {
                        "start": int(start),
                        "length": int(length),
                        "pattern": _decode_key(key, length, base),
                        "freq": freq,
                        "score": int(freq * (length - 1)),
                        "key": key,
                    }
                )
    candidates.sort(key=lambda item: (item["score"], item["freq"], item["length"]), reverse=True)
    selected = []
    seen = set()
    for item in candidates:
        key = (item["start"], item["pattern"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= num_macros:
            break
    return selected


def _macro_index(macro_defs: list[dict[str, object]]) -> dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    by_start: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for macro_id, item in enumerate(macro_defs):
        start = int(item["start"])
        length = int(item["length"])
        by_start.setdefault(start, {}).setdefault(length, []).append((int(item["key"]), macro_id))
    out: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for start, by_len in by_start.items():
        out[start] = {}
        for length, pairs in by_len.items():
            pairs.sort(key=lambda item: item[0])
            keys = torch.tensor([item[0] for item in pairs], dtype=torch.int64)
            ids = torch.tensor([item[1] for item in pairs], dtype=torch.int64)
            out[start][length] = (keys, ids)
    return out


def _compute_program_stats(
    sequence_matrix: torch.Tensor,
    macro_defs: list[dict[str, object]],
    lengths: tuple[int, ...],
    base: int,
    chunk_size: int,
) -> dict[str, object]:
    total_stages = int(sequence_matrix.shape[1])
    macro_idx = _macro_index(macro_defs)
    total_macro_calls = 0
    total_literal_args = 0
    total_macro_tokens = 0
    total_program_units = 0
    total_blocks = int(sequence_matrix.shape[0])
    used_macros: Counter[int] = Counter()
    lengths_sorted = tuple(sorted(lengths, reverse=True))
    for offset in range(0, total_blocks, chunk_size):
        chunk = sequence_matrix[offset:offset + chunk_size]
        rows = int(chunk.shape[0])
        dp_units = [torch.zeros(rows, dtype=torch.int16) for _ in range(total_stages + 1)]
        choice_len = [torch.ones(rows, dtype=torch.int16) for _ in range(total_stages)]
        choice_macro = [torch.full((rows,), -1, dtype=torch.int32) for _ in range(total_stages)]
        key_cache: dict[tuple[int, int], torch.Tensor] = {}
        for start in range(total_stages - 1, -1, -1):
            best_units = (dp_units[start + 1] + 1).clone()
            best_len = torch.ones(rows, dtype=torch.int16)
            best_macro = torch.full((rows,), -1, dtype=torch.int32)
            if start in macro_idx:
                for length in lengths_sorted:
                    if start + length > total_stages or length not in macro_idx[start]:
                        continue
                    cache_key = (start, length)
                    if cache_key not in key_cache:
                        key_cache[cache_key] = _subseq_key_matrix(chunk, start, length, base)
                    keys = key_cache[cache_key]
                    macro_keys, macro_ids = macro_idx[start][length]
                    idx = torch.searchsorted(macro_keys, keys)
                    valid = idx < macro_keys.numel()
                    match = valid & (macro_keys[idx.clamp(max=max(int(macro_keys.numel()) - 1, 0))] == keys)
                    candidate_units = dp_units[start + length] + 1
                    better = match & ((candidate_units < best_units) | ((candidate_units == best_units) & (length > best_len)))
                    if better.any():
                        best_units[better] = candidate_units[better]
                        best_len[better] = int(length)
                        best_macro[better] = macro_ids[idx[better]].to(torch.int32)
            dp_units[start] = best_units
            choice_len[start] = best_len
            choice_macro[start] = best_macro

        current_pos = torch.zeros(rows, dtype=torch.int16)
        alive = torch.ones(rows, dtype=torch.bool)
        while bool(alive.any()):
            active_pos = current_pos[alive].unique(sorted=True)
            for pos in active_pos.tolist():
                rows_mask = alive & (current_pos == pos)
                row_count = int(rows_mask.sum().item())
                if row_count == 0:
                    continue
                macro_ids = choice_macro[pos][rows_mask]
                lens = choice_len[pos][rows_mask].to(torch.int32)
                macro_mask = macro_ids >= 0
                macro_count = int(macro_mask.sum().item())
                total_program_units += row_count
                total_macro_calls += macro_count
                total_literal_args += row_count - macro_count
                if macro_count > 0:
                    macro_ids_cpu = macro_ids[macro_mask].cpu().tolist()
                    for macro_id in macro_ids_cpu:
                        used_macros[int(macro_id)] += 1
                    macro_lengths = lens[macro_mask].sum().item()
                    total_macro_tokens += int(macro_lengths)
                current_pos[rows_mask] = current_pos[rows_mask] + choice_len[pos][rows_mask]
            alive = current_pos < total_stages
    return {
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_macro_calls": float(total_macro_calls / max(total_blocks, 1)),
        "avg_literal_args": float(total_literal_args / max(total_blocks, 1)),
        "macro_token_coverage": float(total_macro_tokens / max(total_blocks * total_stages, 1)),
        "program_compression_ratio": float(total_program_units / max(total_blocks * total_stages, 1)),
        "used_macros": used_macros,
    }


def _build_wal_ss_layer(name: str, enc: GroupedBlockRVQEncoding, args: argparse.Namespace) -> tuple[GroupedBlockRVQEncoding, dict[str, object]]:
    group_rows: list[dict[str, object]] = []
    total_macro_calls = 0
    total_literal_args = 0
    total_macro_tokens = 0
    total_program_units = 0
    total_blocks = 0
    used_macros: Counter[int] = Counter()
    exact_decode = True
    for group_idx, group in enumerate(enc.groups):
        seq_mat = _flatten_group_sequences(group)
        base = max(int(codebook.shape[0]) for codebook in group.codebooks)
        macro_defs = _mine_macros(seq_mat, tuple(sorted(args.macro_lengths)), args.num_macros, base)
        prog_stats = _compute_program_stats(
            seq_mat,
            macro_defs,
            tuple(sorted(args.macro_lengths)),
            base,
            args.chunk_size,
        )
        total_macro_calls += int(round(prog_stats["avg_macro_calls"] * seq_mat.shape[0]))
        total_literal_args += int(round(prog_stats["avg_literal_args"] * seq_mat.shape[0]))
        total_macro_tokens += int(round(prog_stats["macro_token_coverage"] * seq_mat.shape[0] * seq_mat.shape[1]))
        total_program_units += int(round(prog_stats["avg_program_length"] * seq_mat.shape[0]))
        total_blocks += int(seq_mat.shape[0])
        used_macros.update(prog_stats["used_macros"])
        top_macros = [
            {
                "macro_id": idx,
                "start": int(item["start"]),
                "length": int(item["length"]),
                "freq": int(item["freq"]),
                "score": int(item["score"]),
                "used_calls": int(prog_stats["used_macros"][idx]),
            }
            for idx, item in enumerate(macro_defs[:10])
        ]
        group_rows.append(
            {
                "group_idx": int(group_idx),
                "selected_macro_count": int(len(macro_defs)),
                "used_macro_count": int(sum(1 for idx in range(len(macro_defs)) if prog_stats["used_macros"][idx] > 0)),
                "top_macros": top_macros,
            }
        )
    raw_len = len(enc.groups[0].stage_ids) if enc.groups else 0
    stats = {
        "name": name,
        "raw_program_length": int(raw_len),
        "avg_program_length": float(total_program_units / max(total_blocks, 1)),
        "avg_macro_calls": float(total_macro_calls / max(total_blocks, 1)),
        "avg_literal_args": float(total_literal_args / max(total_blocks, 1)),
        "macro_token_coverage": float(total_macro_tokens / max(total_blocks * raw_len, 1)),
        "program_compression_ratio": float(total_program_units / max(total_blocks * raw_len, 1)),
        "selected_macro_count": int(sum(row["selected_macro_count"] for row in group_rows)),
        "used_macro_count": int(sum(row["used_macro_count"] for row in group_rows)),
        "exact_decode": bool(exact_decode),
        "groups": group_rows,
    }
    return enc, stats


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
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--macro-lengths", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--num-macros", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ss_current_l54_gate_up.pt"))
    parser.add_argument("--macro-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ss_macro_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_wal_ss_proto_summary.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    current_cache = Path(args.current_cache)
    current_enc = _build_current_cache(args, current_cache)

    wal_ss_enc: dict[str, GroupedBlockRVQEncoding] = {}
    wal_ss_rows = []
    compare_rows = []
    for name in TARGETS:
        rebuilt, syntax_stats = _build_wal_ss_layer(name, current_enc[name], args)
        wal_ss_enc[name] = rebuilt
        wal_ss_rows.append(syntax_stats)
        compare_rows.append({"name": name, **_compare_encodings(current_enc[name], rebuilt)})
        print(
            f"[wal-ss] {name}: avg_program={syntax_stats['avg_program_length']:.3f}/{syntax_stats['raw_program_length']} "
            f"coverage={syntax_stats['macro_token_coverage']:.3f} used_macros={syntax_stats['used_macro_count']} "
            f"exact={syntax_stats['exact_decode']}",
            flush=True,
        )
    macro_cache = Path(args.macro_cache)
    save_grouped_encoding_map(macro_cache, wal_ss_enc)

    print("\n[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)
    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, current_cache, args.matmul_strategy, args.num_windows)
    macro_eval = dict(current_eval)
    macro_eval["by_construction_identical"] = True

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "macro_lengths": list(args.macro_lengths),
        "num_macros": int(args.num_macros),
        "dense": dense,
        "current_cache": str(current_cache),
        "macro_cache": str(macro_cache),
        "wal_ss": wal_ss_rows,
        "encoding_compare": compare_rows,
        "current_eval": current_eval,
        "macro_eval": macro_eval,
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
        f"  wal_ss_decoded: ppl={macro_eval['metrics']['perplexity']:.4f} tok/s={macro_eval['metrics']['tok_s']:.2f} peak_mb={macro_eval['eval_peak_mb']:.1f}",
        flush=True,
    )
    for row in wal_ss_rows:
        print(
            f"  {row['name']}: avg_program={row['avg_program_length']:.3f}/{row['raw_program_length']} "
            f"compression={row['program_compression_ratio']:.3f} coverage={row['macro_token_coverage']:.3f} "
            f"used_macros={row['used_macro_count']} exact={row['exact_decode']}",
            flush=True,
        )
    print(
        f"  delta(wal_ss-current): ppl={result['delta']['ppl_delta']:+.6f} tok/s={result['delta']['tok_s_delta']:+.2f} peak_mb={result['delta']['peak_mb_delta']:+.1f} (identical by exact decode)",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()