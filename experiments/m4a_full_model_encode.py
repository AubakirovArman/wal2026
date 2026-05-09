"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M4a: encode EVERY linear layer in the model and record relMSE/bpw.

This catches any layer where the recipe fails (e.g., unusual weight distribution)
before committing to full-model PPL tests.
"""
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
from dwl2_dynamic_route.src.codebook import build_codebook  # noqa: E402

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_index():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _open_shard(shard_path):
    from safetensors import safe_open
    return safe_open(shard_path, framework="pt", device="cuda")


def _encode_one(w: torch.Tensor, lmax: int = 12):
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
    wh = decode_routes(enc, ladder, out_dtype=torch.float32) * row_max.float()
    err = rel_mse(w.float(), wh).item()
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max=lmax)
    return err, enc.stop_depth.float().mean().item(), cb.size


def main():
    weight_map = _load_index()
    results = []
    open_shards = {}
    worst = []  # track top-5 worst layers
    t_start = time.time()
    n_done = 0
    for layer in range(80):
        for proj in PROJECTIONS:
            tname = f"model.layers.{layer}.self_attn.{proj}.weight" if proj in ("q_proj","k_proj","v_proj","o_proj") \
                    else f"model.layers.{layer}.mlp.{proj}.weight"
            shard = weight_map.get(tname)
            if shard is None:
                continue
            path = os.path.join(MODEL_DIR, shard)
            if path not in open_shards:
                open_shards[path] = _open_shard(path)
            w = open_shards[path].get_tensor(tname)
            err, avg_depth, cb_size = _encode_one(w, lmax=12)
            results.append({
                "layer": layer, "proj": proj, "shape": list(w.shape),
                "rel_mse": err, "avg_depth": avg_depth, "unique_routes": cb_size,
            })
            worst.append((err, tname, cb_size, avg_depth))
            worst.sort(reverse=True)
            worst = worst[:5]
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t_start
                print(f"  progress {n_done} tensors  elapsed={elapsed:.0f}s")
    # close shards
    for h in open_shards.values():
        h.__exit__(None, None, None)
    # summary
    rels = [r["rel_mse"] for r in results]
    depths = [r["avg_depth"] for r in results]
    routes = [r["unique_routes"] for r in results]
    import statistics as st
    print(f"\n[m4a] total tensors encoded: {len(results)}")
    print(f"  relMSE: mean={st.mean(rels):.2e}  median={st.median(rels):.2e}  "
          f"p90={sorted(rels)[int(0.9*len(rels))]:.2e}  max={max(rels):.2e}")
    print(f"  avg_depth: mean={st.mean(depths):.2f}  median={st.median(depths):.2f}  max={max(depths):.2f}")
    print(f"  unique_routes: mean={st.mean(routes):.0f}  max={max(routes)}")
    print("  worst-5 tensors:")
    for err, name, cb, d in worst:
        print(f"    relMSE={err:.2e}  avg_depth={d:.2f}  routes={cb}  {name}")
    out = "/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m4a_full_model_encode.json"
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"\n[m4a] wrote {out}")


if __name__ == "__main__":
    main()
