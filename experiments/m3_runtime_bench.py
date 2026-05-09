"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M3: correctness + speed benchmark of PackedIDRouteLinear.

For each tensor type in layer 0 we:
  1. load dense weight, row-normalize, calibrate ladder (L_max=12, pinned).
  2. encode → (digits, stop_depth) → build codebook → ids.
  3. wrap into PackedIDRouteLinear.
  4. check that reconstructed weight matches decode_routes * row_max.
  5. benchmark vs dense F.linear on a few (batch × ctx) shapes.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dwl2_dynamic_route.src.route_encoder import decode_routes, encode_routes, rel_mse  # noqa: E402
from dwl2_dynamic_route.src.calibrate import calibrate_ladder  # noqa: E402
from dwl2_dynamic_route.src.codebook import build_codebook  # noqa: E402
from dwl2_dynamic_route.src.runtime import PackedIDRouteLinear  # noqa: E402

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TARGETS = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def _load(tensor_name: str, device: str = "cuda") -> torch.Tensor:
    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(idx_path) as f:
        shard_map = json.load(f)["weight_map"]
    shard = os.path.join(MODEL_DIR, shard_map[tensor_name])
    from safetensors import safe_open
    with safe_open(shard, framework="pt", device=device) as g:
        return g.get_tensor(tensor_name)


def _bench(fn, reps: int = 5, warmup: int = 2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps  # ms


def main():
    results = []
    for tname in TARGETS:
        print(f"\n[m3] {tname}")
        w = _load(tname, device="cuda")
        N, K = w.shape
        lmax = 12
        row_max = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_max
        sample = w_norm.flatten()
        if sample.numel() > 2_000_000:
            idx = torch.randint(0, sample.numel(), (2_000_000,), device=sample.device)
            sample = sample[idx]
        ladder = calibrate_ladder(
            sample, l_max=lmax, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric"
        )
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=lmax)
        cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max=lmax)
        layer = PackedIDRouteLinear.from_encoded(
            ids=ids, codebook_digits=cb.digits, ladder=ladder,
            row_scale=row_max.to(torch.float16),
        ).to("cuda")
        # correctness
        w_rec = layer.reconstruct_weight().float()
        err = rel_mse(w.float(), w_rec).item()
        print(f"  reconstruct relMSE={err:.2e}  unique_routes={cb.size}")
        # benchmark a few shapes
        shapes = [(1, 512), (1, 2048), (4, 2048), (8, 2048)]
        w_dense = w.to(torch.float16).contiguous()
        for B, T in shapes:
            x = torch.randn(B, T, K, device="cuda", dtype=torch.float16)
            ms_route = _bench(lambda: layer(x))
            ms_dense = _bench(lambda: F.linear(x, w_dense))
            ratio = ms_dense / ms_route
            print(
                f"    B={B} T={T:>4}  route={ms_route:7.3f}ms  dense={ms_dense:7.3f}ms  "
                f"route/dense={ms_route/ms_dense:.2f}x  speed_ratio={ratio:.2f}"
            )
            results.append({
                "target": tname, "B": B, "T": T, "ms_route": ms_route, "ms_dense": ms_dense,
                "relMSE": err, "unique_routes": cb.size,
            })
    out_dir = "/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "m3_runtime_bench.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m3] wrote results/m3_runtime_bench.json")


if __name__ == "__main__":
    main()
