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

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.codebook import build_codebook
from dwl2_dynamic_route.src.route_distill import distill_tile_palette, sample_tile_stats
from dwl2_dynamic_route.src.route_encoder import encode_routes
from dwl2_dynamic_route.src.tile_palette_runtime import benchmark_tile_palette_triton_runtime, estimate_local_bpw

TARGETS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def _load_weight(tensor_name: str, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(MODEL_DIR / "model.safetensors.index.json") as handle:
        shard_map = json.load(handle)["weight_map"]
    shard_path = MODEL_DIR / shard_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _distilled_tile(teacher_tile: torch.Tensor, projected_palette: list[float]) -> torch.Tensor:
    palette = torch.tensor(projected_palette, device=teacher_tile.device, dtype=torch.float32)
    assign = (teacher_tile.unsqueeze(-1) - palette.view(1, 1, -1)).abs().argmin(dim=-1)
    return palette[assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-names", nargs="+", default=TARGETS)
    parser.add_argument("--palette-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--sample-tiles", type=int, default=4)
    parser.add_argument("--activation-batch", type=int, default=96)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--bench-tokens", type=int, default=1024)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6e_local_palette_kernel_bench.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records: list[dict[str, object]] = []
    for tensor_idx, tensor_name in enumerate(args.tensor_names):
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
        top_tile = sample_tile_stats(ids, args.tile_rows, args.tile_cols, args.sample_tiles, tensor_idx)[0]
        row1 = top_tile.row0 + args.tile_rows
        col1 = top_tile.col0 + args.tile_cols
        teacher_tile = w_norm[top_tile.row0:row1, top_tile.col0:col1]
        current_tile = routed[top_tile.row0:row1, top_tile.col0:col1]
        tile_ids = ids[top_tile.row0:row1, top_tile.col0:col1]
        for palette_size in args.palette_sizes:
            distilled = distill_tile_palette(
                teacher_tile=teacher_tile,
                current_tile=current_tile,
                candidate_values=codebook_sum,
                palette_size=palette_size,
                activation_batch=args.activation_batch,
                steps=args.steps,
                seed=tensor_idx * 101 + palette_size,
            )
            hard_tile = _distilled_tile(teacher_tile, distilled["projected_palette"])
            bench = benchmark_tile_palette_triton_runtime(tile_ids, codebook_sum, hard_tile, bench_tokens=args.bench_tokens)
            records.append(
                {
                    "tensor_name": tensor_name,
                    "palette_size": palette_size,
                    "base_unique": top_tile.unique_routes,
                    "global_unique_routes": int(codebook.size),
                    "local_bpw_estimate": estimate_local_bpw(int(bench["local_unique"]), hard_tile.numel()),
                    **distilled,
                    **bench,
                }
            )
            print(
                f"[m6e] {tensor_name.split('.')[-3]} p={palette_size:>2} base={top_tile.unique_routes:>4} "
                f"mse={float(distilled['distilled_output_mse']):.3e} "
                f"local/global={float(bench['local_vs_global_triton']):.2f}x "
                f"local/dense={float(bench['local_vs_dense']):.2f}x"
            )

    result = {"records": records}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[m6e] wrote {out_path}")


if __name__ == "__main__":
    main()