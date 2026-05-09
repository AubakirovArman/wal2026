"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-name", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--sample-tiles", type=int, default=12)
    parser.add_argument("--palette-size", type=int, default=16)
    parser.add_argument("--activation-batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6_route_distill_pilot.json"))
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

    tile_stats = sample_tile_stats(ids, args.tile_rows, args.tile_cols, args.sample_tiles, args.seed)
    chosen = tile_stats[0]
    row1 = chosen.row0 + args.tile_rows
    col1 = chosen.col0 + args.tile_cols
    teacher_tile = w_norm[chosen.row0:row1, chosen.col0:col1]
    current_tile = routed[chosen.row0:row1, chosen.col0:col1]
    distilled = distill_tile_palette(
        teacher_tile=teacher_tile,
        current_tile=current_tile,
        candidate_values=codebook_sum,
        palette_size=args.palette_size,
        activation_batch=args.activation_batch,
        steps=args.steps,
        seed=args.seed,
    )
    result = {
        "tensor_name": args.tensor_name,
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "global_unique_routes": int(codebook.size),
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "sample_tiles": tile_stats_to_json(tile_stats),
        "chosen_tile": tile_stats_to_json([chosen])[0],
        "distill": distilled,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[m6] wrote {out_path}")


if __name__ == "__main__":
    main()