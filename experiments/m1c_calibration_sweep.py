"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M1c: sweep calibration modes on mlp.up_proj (the hardest layer-0 tensor)."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dwl2_dynamic_route.src.route_encoder import decode_routes, encode_routes, rel_mse  # noqa: E402
from dwl2_dynamic_route.src.calibrate import calibrate_ladder  # noqa: E402

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TARGETS = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
]


def _load(tensor_name: str, device: str = "cuda") -> torch.Tensor:
    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(idx_path) as f:
        shard_map = json.load(f)["weight_map"]
    shard = os.path.join(MODEL_DIR, shard_map[tensor_name])
    from safetensors import safe_open
    with safe_open(shard, framework="pt", device=device) as g:
        return g.get_tensor(tensor_name)


def _run(name: str, w: torch.Tensor, lmax: int, seed: str, refine_iters: int):
    row_max = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_max
    sample = w_norm.flatten()
    if sample.numel() > 2_000_000:
        idx = torch.randint(0, sample.numel(), (2_000_000,), device=sample.device)
        sample = sample[idx]
    t0 = time.time()
    ladder = calibrate_ladder(
        sample, l_max=lmax, refine_iters=refine_iters, pin_top=True, top_value=1.0, seed=seed
    )
    cal_dt = time.time() - t0
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=lmax)
    wh_norm = decode_routes(enc, ladder, out_dtype=torch.float32)
    w_hat = wh_norm * row_max.float()
    err = rel_mse(w.float(), w_hat).item()
    avg_depth = enc.stop_depth.float().mean().item()
    print(
        f"  seed={seed:>9}  iters={refine_iters:>2}  lmax={lmax}  "
        f"relMSE={err:.6f}  avg_depth={avg_depth:.2f}  cal={cal_dt:.1f}s  "
        f"ladder[:4]={[round(x,4) for x in ladder[:4].tolist()]}"
    )
    return {
        "target": name, "seed": seed, "refine_iters": refine_iters,
        "lmax": lmax, "rel_mse": err, "avg_depth": avg_depth,
        "ladder_head": ladder[:6].tolist(),
    }


def main():
    results = []
    for tname in TARGETS:
        print(f"\n[m1c] {tname}")
        w = _load(tname, device="cuda")
        for seed in ("geometric", "quantile"):
            for iters in (8, 20):
                for lmax in (7, 12):
                    results.append(_run(tname, w, lmax, seed, iters))
    out_dir = "/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m1c_calibration_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m1c] wrote {out_path}")


if __name__ == "__main__":
    main()
