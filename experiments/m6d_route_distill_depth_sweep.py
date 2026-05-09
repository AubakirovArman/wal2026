"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.codebook import build_codebook
from dwl2_dynamic_route.src.route_distill import distill_tile_palette, sample_tile_stats
from dwl2_dynamic_route.src.route_encoder import encode_routes
from dwl2_dynamic_route.src.tile_palette_runtime import benchmark_tile_palette_triton_runtime, estimate_local_bpw


def _load_weight(tensor_name: str, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(MODEL_DIR / "model.safetensors.index.json") as handle:
        shard_map = json.load(handle)["weight_map"]
    shard_path = MODEL_DIR / shard_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _tensor_name(layer_idx: int, family: str) -> str:
    if family in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"model.layers.{layer_idx}.self_attn.{family}.weight"
    return f"model.layers.{layer_idx}.mlp.{family}.weight"


def _distilled_tile(teacher_tile: torch.Tensor, projected_palette: list[float]) -> torch.Tensor:
    palette = torch.tensor(projected_palette, device=teacher_tile.device, dtype=torch.float32)
    assign = (teacher_tile.unsqueeze(-1) - palette.view(1, 1, -1)).abs().argmin(dim=-1)
    return palette[assign]


def _summarize(entries: list[dict[str, object]]) -> dict[str, float]:
    return {
        "count": float(len(entries)),
        "mean_base_unique": float(statistics.mean(int(item["base_unique"]) for item in entries)),
        "mean_distilled_output_mse": float(statistics.mean(float(item["distilled_output_mse"]) for item in entries)),
        "mean_local_bpw": float(statistics.mean(float(item["local_bpw_estimate"]) for item in entries)),
        "mean_local_vs_global_triton": float(statistics.mean(float(item["local_vs_global_triton"]) for item in entries)),
        "mean_local_vs_dense": float(statistics.mean(float(item["local_vs_dense"]) for item in entries)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-indices", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79])
    parser.add_argument("--families", nargs="+", default=["q_proj", "up_proj"])
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--sample-tiles", type=int, default=4)
    parser.add_argument("--palette-size", type=int, default=32)
    parser.add_argument("--activation-batch", type=int, default=96)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--bench-tokens", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6d_route_distill_depth_sweep.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records: list[dict[str, object]] = []
    for family in args.families:
        for layer_idx in args.layer_indices:
            tensor_name = _tensor_name(layer_idx, family)
            weight = _load_weight(tensor_name, device=device)
            row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            w_norm = weight / row_scale
            sample = w_norm.flatten()
            if sample.numel() > args.sample_limit:
                idx = torch.randint(0, sample.numel(), (args.sample_limit,), device=sample.device)
                sample = sample[idx]
            ladder = calibrate_ladder(sample, l_max=12, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric")
            enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=12)
            codebook, ids = build_codebook(enc.digits, enc.stop_depth, l_max=12)
            codebook_sum = (codebook.digits.to(torch.float32) * ladder.to(torch.float32)).sum(dim=-1)
            routed = codebook_sum[ids.long()]
            top_tile = sample_tile_stats(ids, args.tile_rows, args.tile_cols, args.sample_tiles, layer_idx)[0]
            row1 = top_tile.row0 + args.tile_rows
            col1 = top_tile.col0 + args.tile_cols
            teacher_tile = w_norm[top_tile.row0:row1, top_tile.col0:col1]
            current_tile = routed[top_tile.row0:row1, top_tile.col0:col1]
            tile_ids = ids[top_tile.row0:row1, top_tile.col0:col1]
            distilled = distill_tile_palette(
                teacher_tile=teacher_tile,
                current_tile=current_tile,
                candidate_values=codebook_sum,
                palette_size=args.palette_size,
                activation_batch=args.activation_batch,
                steps=args.steps,
                seed=layer_idx + len(records) * 17 + args.palette_size,
            )
            hard_tile = _distilled_tile(teacher_tile, distilled["projected_palette"])
            bench = benchmark_tile_palette_triton_runtime(
                tile_ids,
                codebook_sum,
                hard_tile,
                bench_tokens=args.bench_tokens,
            )
            record = {
                "family": family,
                "layer_idx": layer_idx,
                "tensor_name": tensor_name,
                "base_unique": top_tile.unique_routes,
                "tile_row0": top_tile.row0,
                "tile_col0": top_tile.col0,
                "global_unique_routes": int(codebook.size),
                "local_bpw_estimate": estimate_local_bpw(int(bench["local_unique"]), hard_tile.numel()),
                **distilled,
                **bench,
            }
            records.append(record)
            print(
                f"[m6d] {family} layer={layer_idx:>2} base={top_tile.unique_routes:>4} "
                f"mse={float(record['distilled_output_mse']):.3e} "
                f"local/global={float(record['local_vs_global_triton']):.2f}x "
                f"local/dense={float(record['local_vs_dense']):.2f}x"
            )

    by_family = {
        family: _summarize([item for item in records if item["family"] == family])
        for family in args.families
    }
    result = {
        "layer_indices": args.layer_indices,
        "families": args.families,
        "palette_size": args.palette_size,
        "records": records,
        "summary": by_family,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"[m6d] wrote {out_path}")


if __name__ == "__main__":
    main()