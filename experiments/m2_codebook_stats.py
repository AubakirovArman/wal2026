"""M2: count unique routes per layer-0 tensor and compute effective bpw."""
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
TARGETS = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
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


def _run(name: str, w: torch.Tensor, lmax: int):
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
    # verify decode
    wh = decode_routes(enc, ladder, out_dtype=torch.float32) * row_max.float()
    err = rel_mse(w.float(), wh).item()

    t0 = time.time()
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max=lmax)
    dt = time.time() - t0

    n_weights = w.numel()
    M = cb.size
    id_bits = math.ceil(math.log2(max(M, 2)))
    bpw_ids = id_bits                                 # per weight (one ID per position)
    # codebook overhead: M entries × (L_max ternary digits + 4-bit stop) ≈ L_max * log2(3) + 4 bits
    cb_bits = M * (lmax * math.log2(3) + 4)
    bpw_cb = cb_bits / n_weights
    # row_scale: 16 bits per row
    rows = w.shape[0]
    bpw_row = 16.0 * rows / n_weights
    total_bpw = bpw_ids + bpw_cb + bpw_row

    print(
        f"  {name.split('.')[-2]:>10}  shape={tuple(w.shape)}  "
        f"unique_routes={M:>8,}  id_bits={id_bits}  "
        f"bpw(ids)={bpw_ids}  bpw(cb)={bpw_cb:.2f}  bpw(row)={bpw_row:.3f}  "
        f"TOTAL={total_bpw:.2f}  relMSE={err:.2e}  build={dt:.2f}s"
    )
    return {
        "target": name, "lmax": lmax, "shape": list(w.shape),
        "unique_routes": M, "id_bits": id_bits,
        "bpw_ids": bpw_ids, "bpw_cb": bpw_cb, "bpw_row": bpw_row, "bpw_total": total_bpw,
        "rel_mse": err,
    }


def main():
    results = []
    for tname in TARGETS:
        print(f"\n[m2] {tname}")
        w = _load(tname, device="cuda")
        for lmax in (7, 9, 12):
            results.append(_run(tname, w, lmax))
    out_dir = "/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m2_codebook_stats.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m2] wrote {out_path}")


if __name__ == "__main__":
    main()
