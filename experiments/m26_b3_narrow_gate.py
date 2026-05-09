"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import encode_grouped_block_residual_vq
from dwl2_dynamic_route.src.encoding_io import load_grouped_encoding_map, save_grouped_encoding_map
from dwl2_dynamic_route.src.runtime import PackedGroupedBlockRVQLinear


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TARGETS = ("model.layers.54.mlp.gate_proj", "model.layers.54.mlp.up_proj")


def _load_or_build_cache(args: argparse.Namespace, cache_path: Path) -> dict[str, object]:
    if cache_path.exists() and not args.rebuild_cache:
        return load_grouped_encoding_map(cache_path)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    encodings = {}
    target_set = set(TARGETS)
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        encodings[name] = encode_grouped_block_residual_vq(
            module.weight.detach(),
            group_rows=args.group_rows,
            block_size=args.block_size,
            codebook_size=args.codebook_size,
            num_stages=args.num_stages,
            product_splits=args.product_splits,
            normalize_blocks=args.normalize_blocks,
            transform_kind=args.transform_kind,
            calibrate_stage_scales=args.calibrate_stage_scales,
            residual_correction=args.residual_correction,
            sample_limit=args.sample_limit,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
    save_grouped_encoding_map(cache_path, encodings)
    return encodings


def _make_module(enc, strategy: str, args: argparse.Namespace, device: torch.device) -> PackedGroupedBlockRVQLinear:
    mod = PackedGroupedBlockRVQLinear(
        enc,
        bias=None,
        matmul_strategy=strategy,
        hot_topk=args.hot_topk,
        hot_score_mode=args.hot_score_mode,
        hot_min_stage_share=args.hot_min_stage_share,
        hot_score_threshold_ratio=args.hot_score_threshold_ratio,
    ).to(device)
    mod.eval()
    return mod


def _bench_ms(mod: nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mod(x)
    torch.cuda.synchronize(x.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        mod(x)
    end.record()
    torch.cuda.synchronize(x.device)
    return float(start.elapsed_time(end) / max(iters, 1))


def _onehot_max_abs(ref_mod: nn.Module, test_mod: nn.Module, in_features: int, chunk: int, device: torch.device) -> float:
    max_abs = 0.0
    for start in range(0, in_features, chunk):
        cols = min(chunk, in_features - start)
        x = torch.zeros((cols, in_features), dtype=torch.bfloat16, device=device)
        pos = torch.arange(cols, device=device)
        x[pos, start + pos] = 1
        with torch.no_grad():
            y_ref = ref_mod(x)
            y_test = test_mod(x)
        max_abs = max(max_abs, float((y_ref.float() - y_test.float()).abs().max().item()))
    return max_abs


def _hot_hit_stats(mod: PackedGroupedBlockRVQLinear, args: argparse.Namespace) -> tuple[float, float]:
    rates = []
    for group in mod.groups:
        group._build_stage_hot_cache(
            torch.bfloat16,
            group.row_scale.device,
            int(args.hot_topk),
            args.hot_score_mode,
            float(args.hot_min_stage_share),
            float(args.hot_score_threshold_ratio),
        )
        for idx in group._compute_active_stage_indices():
            ids = getattr(group, f"stage_ids_{idx}")
            hot_pos = None if group._hot_positions_cached is None else group._hot_positions_cached[idx]
            rates.append(0.0 if hot_pos is None else float(hot_pos.numel()) / max(int(ids.numel()), 1))
    return (sum(rates) / max(len(rates), 1), min(rates) if rates else 0.0)


def _run_target(name: str, enc, args: argparse.Namespace, device: torch.device) -> dict[str, float | bool | str]:
    ref_mod = _make_module(enc, "full_weight_hot_v2", args, device)
    test_mod = _make_module(enc, "stage_local_hot_cold_b3", args, device)
    in_features = int(enc.original_shape[1])
    x = torch.randn((args.m, in_features), dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        y_ref = ref_mod(x)
        y_test = test_mod(x)
    diff = (y_ref.float() - y_test.float()).abs()
    ms_hot_v2 = _bench_ms(ref_mod, x, args.warmup, args.iters)
    ms_b3 = _bench_ms(test_mod, x, args.warmup, args.iters)
    hit_mean, hit_min = _hot_hit_stats(test_mod, args)
    staged_kb = int(args.hot_topk) * int(test_mod.groups[0].block_size) * 2 / 1024.0
    speedup = ms_hot_v2 / max(ms_b3, 1e-12)
    result = {
        "target": name,
        "m": int(args.m),
        "onehot_max_abs_diff": _onehot_max_abs(ref_mod, test_mod, in_features, args.onehot_chunk, device),
        "rand_max_abs_diff": float(diff.max().item()),
        "rand_mean_abs_diff": float(diff.mean().item()),
        "has_nan": bool(torch.isnan(y_test).any().item()),
        "ms_hot_v2": ms_hot_v2,
        "ms_b3": ms_b3,
        "speedup_vs_hot_v2": speedup,
        "hot_hit_rate_mean": hit_mean,
        "hot_hit_rate_min": hit_min,
        "staged_hot_kb_per_stage": staged_kb,
    }
    result["pass_narrow_gate"] = bool(
        result["onehot_max_abs_diff"] < 5e-3
        and not result["has_nan"]
        and speedup >= float(args.min_speedup)
        and hit_mean >= float(args.min_hot_hit_rate_mean)
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", default=str(ROOT / "dwl2_dynamic_route/results/m25_l54_q_gu_encodings.pt"))
    ap.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m26_b3_narrow_gate.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--onehot-chunk", type=int, default=256)
    ap.add_argument("--hot-topk", type=int, default=64)
    ap.add_argument("--hot-score-mode", choices=("count", "row_scale_norm", "stage_influence_proxy"), default="count")
    ap.add_argument("--hot-min-stage-share", type=float, default=0.0)
    ap.add_argument("--hot-score-threshold-ratio", type=float, default=0.65)
    ap.add_argument("--min-hot-hit-rate-mean", type=float, default=0.70)
    ap.add_argument("--min-speedup", type=float, default=1.20)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--group-rows", type=int, default=28672)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    ap.add_argument("--normalize-blocks", choices=("none", "amax", "l2"), default="none")
    ap.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default="none")
    ap.add_argument("--calibrate-stage-scales", action="store_true")
    ap.add_argument("--residual-correction", choices=("none", "sign"), default="none")
    ap.add_argument("--sample-limit", type=int, default=65536)
    ap.add_argument("--kmeans-iters", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16384)
    args = ap.parse_args()

    device = torch.device(args.device)
    cache_path = Path(args.cache_path)
    encodings = _load_or_build_cache(args, cache_path)
    results = {name: _run_target(name, encodings[name], args, device) for name in TARGETS}
    results["all_pass_narrow_gate"] = all(bool(item["pass_narrow_gate"]) for item in results.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    for name, item in results.items():
        if not isinstance(item, dict):
            continue
        print(
            f"{name}: onehot={item['onehot_max_abs_diff']:.6f} has_nan={item['has_nan']} "
            f"hot_v2={item['ms_hot_v2']:.3f}ms b3={item['ms_b3']:.3f}ms speedup={item['speedup_vs_hot_v2']:.3f} "
            f"hit_mean={item['hot_hit_rate_mean']:.3f} hit_min={item['hot_hit_rate_min']:.3f} pass={item['pass_narrow_gate']}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()