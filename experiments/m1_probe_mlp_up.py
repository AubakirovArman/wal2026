"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""M1 probe: encode one mlp.up_proj tensor of Llama 3.3 70B with dynamic-depth routes.

Loads the single weight from the HF safetensors shard using safetensors' lazy API
so we do not pay for the full model in memory. Calibrates a family ladder on a
random weight sample, encodes the full tensor at several L_max and stop_threshold
settings, and prints relMSE + average depth + implied bits/weight (digits only).

Usage (from project root):
    CUDA_VISIBLE_DEVICES=2 venv/bin/python dwl2_dynamic_route/experiments/m1_probe_mlp_up.py

No CUDA is strictly required — falls back to CPU.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # make `dwl2_dynamic_route` importable

from dwl2_dynamic_route.src.calibrate import calibrate_ladder  # noqa: E402
from dwl2_dynamic_route.src.route_encoder import (  # noqa: E402
    decode_routes,
    encode_routes,
    rel_mse,
)

MODEL_SNAPSHOT = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
)
TARGET_NAME = "model.layers.0.mlp.up_proj.weight"
SAMPLE_LIMIT = 2_000_000  # for ladder calibration


def _find_shard_for(tensor_name: str) -> Path:
    index = json.loads((MODEL_SNAPSHOT / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][tensor_name]
    return MODEL_SNAPSHOT / shard


def _load_tensor(tensor_name: str, device: str) -> torch.Tensor:
    shard = _find_shard_for(tensor_name)
    with safe_open(str(shard), framework="pt", device=device) as f:
        w = f.get_tensor(tensor_name)
    return w


def _summarize(name: str, w: torch.Tensor, lmax: int, stop_thr: float):
    ladder = calibrate_ladder(w.flatten()[:SAMPLE_LIMIT], l_max=lmax)
    t0 = time.time()
    enc = encode_routes(w, ladder, stop_threshold=stop_thr, l_max=lmax)
    w_hat = decode_routes(enc, ladder, out_dtype=torch.float32)
    dt = time.time() - t0
    err = rel_mse(w.float(), w_hat).item()
    avg_depth = enc.stop_depth.to(torch.float32).mean().item()
    # bits per weight for digits only (naive): ~log2(3) per used digit
    import math

    bpw_digits = avg_depth * math.log2(3.0)
    nonzero_frac = (enc.digits != 0).to(torch.float32).mean().item() * lmax
    print(
        f"{name:<32} lmax={lmax:>2} stop={stop_thr:<7}  "
        f"relMSE={err:.6f}  avg_depth={avg_depth:.2f}  "
        f"bpw(dig)~{bpw_digits:.2f}  enc_time={dt:.2f}s"
    )
    return {
        "name": name,
        "lmax": lmax,
        "stop_threshold": stop_thr,
        "rel_mse": err,
        "avg_depth": avg_depth,
        "bpw_digits": bpw_digits,
        "ladder": ladder.tolist(),
        "encode_time_s": dt,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[m1_probe] loading {TARGET_NAME} on {device} ...")
    w = _load_tensor(TARGET_NAME, device=device).to(torch.float32)
    print(
        f"[m1_probe] shape={tuple(w.shape)}  |w|max={w.abs().max().item():.4f}  "
        f"|w|mean={w.abs().mean().item():.5f}  device={w.device}"
    )
    out = []
    for lmax in (5, 7, 9, 12):
        for stop_thr in (0.0, 0.002, 0.001):
            out.append(_summarize(f"L{lmax}-t{stop_thr}", w, lmax=lmax, stop_thr=stop_thr))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "m1_probe_mlp_up.json").write_text(json.dumps(out, indent=2))
    print(f"[m1_probe] wrote {results_dir / 'm1_probe_mlp_up.json'}")


if __name__ == "__main__":
    main()
