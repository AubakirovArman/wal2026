"""M27 FGRL Step 3a: finer-grained route-language re-encode for l54.gate/up.

This script tests the user's "change the language, not the selector" idea on
the exact slice where M26 B0-B3, RRF, and PTDP all failed:

  - model.language_model.layers.54.mlp.gate_proj
  - model.language_model.layers.54.mlp.up_proj

It compares two encodings on the same two-layer slice:

  1. current packed config: total 12 stages, codebook 256
  2. FGRL candidate:       total 20 stages, codebook 80

Important implementation note:
the repo's encoder parameter ``num_stages`` is *per product split*.
So to realize a total depth of 20 while keeping ``product_splits=4`` we use
``stages_per_split=(5, 5, 5, 5)``.

Outputs:
  - per-layer static metrics: relMSE, bits/weight, avg tile occupancy,
    avg unique ids per tile, top64 share inside tiles
  - slice-level eval metrics: PPL, tok/s, peak VRAM on the same text windows
  - cached encodings for the current and FGRL variants
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
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


def _eval_ids(tok: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


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


def _rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.float() - b.float()).square().mean()
    den = a.float().square().mean().clamp_min(1e-12)
    return float((num / den).item())


def _load_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


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


def _balanced_stages_per_split(total_stages: int, product_splits: int) -> tuple[int, ...]:
    if total_stages < product_splits:
        raise ValueError("total_stages must be >= product_splits")
    base = total_stages // product_splits
    rem = total_stages % product_splits
    return tuple(base + (1 if split_idx < rem else 0) for split_idx in range(product_splits))


def _tile_metrics(
    enc: GroupedBlockRVQEncoding,
    row_tile_size: int,
    col_tile_size: int,
    topk: int,
) -> dict[str, float]:
    occupancy_values: list[float] = []
    unique_values: list[float] = []
    topk_share_values: list[float] = []
    topk_share_ge_075 = 0
    total_tiles = 0
    for group in enc.groups:
        block_cols_per_tile = col_tile_size // int(group.block_size)
        if block_cols_per_tile < 1:
            raise ValueError("col_tile_size must be at least one full block")
        rows, blocks_per_row = group.stage_shape
        num_row_tiles = math.ceil(rows / row_tile_size)
        num_col_tiles = math.ceil(blocks_per_row / block_cols_per_tile)
        for stage_ids in group.stage_ids:
            stage_ids_cpu = stage_ids.cpu().to(torch.int64)
            codebook_size = int(stage_ids_cpu.max().item()) + 1
            for row_tile_idx in range(num_row_tiles):
                row_a = row_tile_idx * row_tile_size
                row_b = min(row_a + row_tile_size, rows)
                for col_tile_idx in range(num_col_tiles):
                    block_a = col_tile_idx * block_cols_per_tile
                    block_b = min(block_a + block_cols_per_tile, blocks_per_row)
                    ids = stage_ids_cpu[row_a:row_b, block_a:block_b].reshape(-1)
                    counts = torch.bincount(ids, minlength=codebook_size).float()
                    unique_ct = int((counts > 0).sum().item())
                    share = float(torch.topk(counts, k=min(topk, codebook_size)).values.sum() / counts.sum().clamp_min(1.0))
                    occupancy_values.append(unique_ct / float(codebook_size))
                    unique_values.append(float(unique_ct))
                    topk_share_values.append(share)
                    topk_share_ge_075 += int(share >= 0.75)
                    total_tiles += 1
    return {
        "avg_tile_occupancy": float(sum(occupancy_values) / max(len(occupancy_values), 1)),
        "avg_unique_ids_per_tile": float(sum(unique_values) / max(len(unique_values), 1)),
        f"mean_top{topk}_share": float(sum(topk_share_values) / max(len(topk_share_values), 1)),
        f"median_top{topk}_share": float(torch.tensor(topk_share_values).median().item() if topk_share_values else 0.0),
        f"frac_top{topk}_share_ge_075": float(topk_share_ge_075 / max(total_tiles, 1)),
        "num_stage_tiles": int(total_tiles),
    }


def _encode_variant(
    model: AutoModelForCausalLM,
    label: str,
    args: argparse.Namespace,
    cache_path: Path,
    *,
    codebook_size: int,
    stages_per_split: tuple[int, ...],
) -> dict[str, object]:
    print(f"\n[encode:{label}] codebook={codebook_size} stages_per_split={stages_per_split}", flush=True)
    encodings: dict[str, GroupedBlockRVQEncoding] = {}
    layer_rows: list[dict[str, object]] = []
    total_stage_tensors = sum(stages_per_split)
    module_map = dict(model.named_modules())
    for name in TARGETS:
        module = module_map[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name} is not nn.Linear")
        t0 = time.time()
        enc = encode_grouped_block_residual_vq(
            module.weight.detach(),
            group_rows=args.group_rows,
            block_size=args.block_size,
            codebook_size=codebook_size,
            num_stages=stages_per_split[0],
            product_splits=args.product_splits,
            stages_per_split=stages_per_split,
            normalize_blocks=args.normalize_blocks,
            transform_kind=args.transform_kind,
            calibrate_stage_scales=args.calibrate_stage_scales,
            residual_correction=args.residual_correction,
            sample_limit=args.sample_limit,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
        recon = enc.reconstruct(out_dtype=torch.bfloat16).to(module.weight.device)
        tile = _tile_metrics(enc, args.row_tile_size, args.col_tile_size, args.topk)
        row = {
            "name": name,
            "codebook_size": int(codebook_size),
            "product_splits": int(args.product_splits),
            "stages_per_split": list(stages_per_split),
            "total_stage_tensors": int(total_stage_tensors),
            "rel_mse": _rel_mse(module.weight.detach(), recon),
            "sample_rel_mse": float(enc.sample_rel_mse),
            "bits_per_weight": float(enc.bits_per_weight()),
            **tile,
            "encode_s": time.time() - t0,
        }
        encodings[name] = enc
        layer_rows.append(row)
        print(
            f"  {name}: rel_mse={row['rel_mse']:.3e} bits/w={row['bits_per_weight']:.3f} "
            f"avg_occ={row['avg_tile_occupancy']:.3f} avg_unique={row['avg_unique_ids_per_tile']:.1f} "
            f"top{args.topk}_share={row[f'mean_top{args.topk}_share']:.3f} ({row['encode_s']:.1f}s)",
            flush=True,
        )
        del recon
    save_grouped_encoding_map(cache_path, encodings)
    aggregate = {
        "avg_rel_mse": float(sum(item["rel_mse"] for item in layer_rows) / max(len(layer_rows), 1)),
        "avg_sample_rel_mse": float(sum(item["sample_rel_mse"] for item in layer_rows) / max(len(layer_rows), 1)),
        "avg_bits_per_weight": float(sum(item["bits_per_weight"] for item in layer_rows) / max(len(layer_rows), 1)),
        "avg_tile_occupancy": float(sum(item["avg_tile_occupancy"] for item in layer_rows) / max(len(layer_rows), 1)),
        "avg_unique_ids_per_tile": float(sum(item["avg_unique_ids_per_tile"] for item in layer_rows) / max(len(layer_rows), 1)),
        f"mean_top{args.topk}_share": float(sum(item[f"mean_top{args.topk}_share"] for item in layer_rows) / max(len(layer_rows), 1)),
        f"mean_frac_top{args.topk}_share_ge_075": float(sum(item[f"frac_top{args.topk}_share_ge_075"] for item in layer_rows) / max(len(layer_rows), 1)),
    }
    return {
        "label": label,
        "cache_path": str(cache_path),
        "codebook_size": int(codebook_size),
        "stages_per_split": list(stages_per_split),
        "total_stage_tensors": int(total_stage_tensors),
        "layer_stats": layer_rows,
        "aggregate": aggregate,
    }


def _run_preencoded_eval(
    ids: torch.Tensor,
    cache_path: Path,
    strategy: str,
    num_windows: int,
) -> dict[str, object]:
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
    return {
        "metrics": metrics,
        "eval_vram": eval_vram,
        "eval_peak_mb": _peak(eval_vram),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-source", choices=("raw", "local"), default="local")
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--current-total-stages", type=int, default=12)
    parser.add_argument("--current-codebook-size", type=int, default=256)
    parser.add_argument("--fgrl-total-stages", type=int, default=20)
    parser.add_argument("--fgrl-codebook-size", type=int, default=80)
    parser.add_argument("--row-tile-size", type=int, default=64)
    parser.add_argument("--col-tile-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--normalize-blocks", choices=("none", "amax", "l2"), default="none")
    parser.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default="none")
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--residual-correction", choices=("none", "sign"), default="none")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--matmul-strategy", default="full_weight_fast")
    parser.add_argument("--current-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_fgrl_current_l54_gate_up.pt"))
    parser.add_argument("--fgrl-cache", default=str(ROOT / "dwl2_dynamic_route/results/m27_fgrl_candidate_l54_gate_up.pt"))
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m27_fgrl_reencode_summary.json"))
    args = parser.parse_args()

    current_sps = _balanced_stages_per_split(args.current_total_stages, args.product_splits)
    fgrl_sps = _balanced_stages_per_split(args.fgrl_total_stages, args.product_splits)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    print("[dense] evaluate untouched model", flush=True)
    dense = _run_dense_eval(ids, args.num_windows)

    print("\n[encode] load model once and build both caches", flush=True)
    model = _load_model()
    current = _encode_variant(
        model,
        "current_12x256",
        args,
        Path(args.current_cache),
        codebook_size=args.current_codebook_size,
        stages_per_split=current_sps,
    )
    fgrl = _encode_variant(
        model,
        "fgrl_20x80",
        args,
        Path(args.fgrl_cache),
        codebook_size=args.fgrl_codebook_size,
        stages_per_split=fgrl_sps,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[eval] current via {args.matmul_strategy}", flush=True)
    current_eval = _run_preencoded_eval(ids, Path(args.current_cache), args.matmul_strategy, args.num_windows)
    print(f"\n[eval] fgrl via {args.matmul_strategy}", flush=True)
    fgrl_eval = _run_preencoded_eval(ids, Path(args.fgrl_cache), args.matmul_strategy, args.num_windows)

    result = {
        "targets": list(TARGETS),
        "text_source": args.text_source,
        "num_windows": int(args.num_windows),
        "matmul_strategy": args.matmul_strategy,
        "row_tile_size": int(args.row_tile_size),
        "col_tile_size": int(args.col_tile_size),
        "topk": int(args.topk),
        "dense": dense,
        "current": {**current, "eval": current_eval},
        "fgrl": {**fgrl, "eval": fgrl_eval},
        "delta": {
            "tile_occupancy_drop": float(current["aggregate"]["avg_tile_occupancy"] - fgrl["aggregate"]["avg_tile_occupancy"]),
            f"top{args.topk}_share_gain": float(fgrl["aggregate"][f"mean_top{args.topk}_share"] - current["aggregate"][f"mean_top{args.topk}_share"]),
            "avg_rel_mse_delta": float(fgrl["aggregate"]["avg_rel_mse"] - current["aggregate"]["avg_rel_mse"]),
            "ppl_delta": float(fgrl_eval["metrics"]["perplexity"] - current_eval["metrics"]["perplexity"]),
            "tok_s_delta": float(fgrl_eval["metrics"]["tok_s"] - current_eval["metrics"]["tok_s"]),
            "peak_mb_delta": float(fgrl_eval["eval_peak_mb"] - current_eval["eval_peak_mb"]),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===")
    print(
        f"  dense:       ppl={dense['metrics']['perplexity']:.4f} tok/s={dense['metrics']['tok_s']:.2f} peak_mb={dense['eval_peak_mb']:.1f}",
        flush=True,
    )
    print(
        f"  current:     ppl={current_eval['metrics']['perplexity']:.4f} tok/s={current_eval['metrics']['tok_s']:.2f} peak_mb={current_eval['eval_peak_mb']:.1f} "
        f"avg_occ={current['aggregate']['avg_tile_occupancy']:.3f} top{args.topk}_share={current['aggregate'][f'mean_top{args.topk}_share']:.3f} "
        f"avg_rel_mse={current['aggregate']['avg_rel_mse']:.3e}",
        flush=True,
    )
    print(
        f"  fgrl:        ppl={fgrl_eval['metrics']['perplexity']:.4f} tok/s={fgrl_eval['metrics']['tok_s']:.2f} peak_mb={fgrl_eval['eval_peak_mb']:.1f} "
        f"avg_occ={fgrl['aggregate']['avg_tile_occupancy']:.3f} top{args.topk}_share={fgrl['aggregate'][f'mean_top{args.topk}_share']:.3f} "
        f"avg_rel_mse={fgrl['aggregate']['avg_rel_mse']:.3e}",
        flush=True,
    )
    print(
        f"  delta(fgrl-current): occ={result['delta']['tile_occupancy_drop']:+.3f} "
        f"top{args.topk}_share={result['delta'][f'top{args.topk}_share_gain']:+.3f} "
        f"ppl={result['delta']['ppl_delta']:+.4f} tok/s={result['delta']['tok_s_delta']:+.2f} peak_mb={result['delta']['peak_mb_delta']:+.1f}",
        flush=True,
    )
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()