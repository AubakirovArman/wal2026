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
from dwl2_dynamic_route.src.route_distill import distill_tile_palette, sample_tile_stats, tile_stats_to_json
from dwl2_dynamic_route.src.route_encoder import encode_routes


def _load_weight(tensor_name: str, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(MODEL_DIR / "model.safetensors.index.json") as handle:
        shard_map = json.load(handle)["weight_map"]
    shard_path = MODEL_DIR / shard_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _summarize(entries: list[dict[str, object]]) -> dict[str, float | int]:
    out_mse = [float(item["distilled_output_mse"]) for item in entries]
    wt_mse = [float(item["distilled_weight_mse"]) for item in entries]
    uniq = [int(item["distilled_unique"]) for item in entries]
    return {
        "tiles": len(entries),
        "mean_distilled_output_mse": float(statistics.mean(out_mse)),
        "median_distilled_output_mse": float(statistics.median(out_mse)),
        "mean_distilled_weight_mse": float(statistics.mean(wt_mse)),
        "median_distilled_weight_mse": float(statistics.median(wt_mse)),
        "mean_distilled_unique": float(statistics.mean(uniq)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-name", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--sample-tiles", type=int, default=8)
    parser.add_argument("--use-top-tiles", type=int, default=4)
    parser.add_argument("--palette-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--activation-batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6b_route_distill_sweep.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = _load_weight(args.tensor_name, device=device)
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

    sampled_tiles = sample_tile_stats(ids, args.tile_rows, args.tile_cols, args.sample_tiles, args.seed)
    selected_tiles = sampled_tiles[: min(args.use_top_tiles, len(sampled_tiles))]
    sweep: dict[str, list[dict[str, object]]] = {str(size): [] for size in args.palette_sizes}
    for tile_idx, tile in enumerate(selected_tiles):
        row1 = tile.row0 + args.tile_rows
        col1 = tile.col0 + args.tile_cols
        teacher_tile = w_norm[tile.row0:row1, tile.col0:col1]
        current_tile = routed[tile.row0:row1, tile.col0:col1]
        for palette_size in args.palette_sizes:
            distilled = distill_tile_palette(
                teacher_tile=teacher_tile,
                current_tile=current_tile,
                candidate_values=codebook_sum,
                palette_size=palette_size,
                activation_batch=args.activation_batch,
                steps=args.steps,
                seed=args.seed + tile_idx * 17 + palette_size,
            )
            sweep[str(palette_size)].append(
                {
                    "row0": tile.row0,
                    "col0": tile.col0,
                    "base_unique": tile.unique_routes,
                    "base_usage_entropy_bits": tile.usage_entropy_bits,
                    **distilled,
                }
            )
    summary = {key: _summarize(entries) for key, entries in sweep.items()}
    result = {
        "tensor_name": args.tensor_name,
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "global_unique_routes": int(codebook.size),
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "sampled_tiles": tile_stats_to_json(sampled_tiles),
        "selected_tiles": tile_stats_to_json(selected_tiles),
        "palette_sizes": args.palette_sizes,
        "summary": summary,
        "sweep": sweep,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[m6b] wrote {out_path}")


if __name__ == "__main__":
    main()