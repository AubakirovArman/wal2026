"""M1 probe with per-row normalization (mandatory for real LLM weights).

Strategy mirrors Route B's row_scale:
    row_max[n]   = max(|W[n,:]|) clamp_min(eps)
    W_norm[n,k]  = W[n,k] / row_max[n]            # max |.| == 1 per row
    encode W_norm with a single ladder seeded near 1.0
    decode W_hat = W_norm_hat * row_max[:, None]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder  # noqa: E402
from dwl2_dynamic_route.src.route_encoder import (  # noqa: E402
    decode_routes,
    encode_routes,
    rel_mse,
)

SNAPSHOT = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
)
TARGETS = [
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def _load_tensor(name: str, device: str) -> torch.Tensor:
    idx = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    shard = SNAPSHOT / idx["weight_map"][name]
    with safe_open(str(shard), framework="pt", device=device) as f:
        return f.get_tensor(name)


def _run(name: str, w: torch.Tensor, lmax: int):
    row_max = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_max  # max |w_norm[n,:]| == 1 exactly
    sample = w_norm.flatten()
    if sample.numel() > 2_000_000:
        idx = torch.randint(0, sample.numel(), (2_000_000,), device=sample.device)
        sample = sample[idx]
    ladder = calibrate_ladder(sample, l_max=lmax, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric")
    t0 = time.time()
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=lmax)
    wh_norm = decode_routes(enc, ladder, out_dtype=torch.float32)
    w_hat = wh_norm * row_max.float()
    dt = time.time() - t0
    err = rel_mse(w.float(), w_hat).item()
    avg_depth = enc.stop_depth.float().mean().item()
    bpw_digits = avg_depth * math.log2(3.0)
    k = w.shape[-1]
    bpw_rowscale = 16.0 / k
    print(
        f"  lmax={lmax:>2}  relMSE={err:.6f}  avg_depth={avg_depth:.2f}  "
        f"bpw(dig)~{bpw_digits:.2f}  +rowscale={bpw_rowscale:.3f}  enc={dt:.2f}s  "
        f"ladder[:4]={[round(x,4) for x in ladder[:4].tolist()]}"
    )
    return {
        "target": name,
        "lmax": lmax,
        "rel_mse": err,
        "avg_depth": avg_depth,
        "bpw_digits": bpw_digits,
        "bpw_rowscale": bpw_rowscale,
        "ladder_head": ladder[:6].tolist(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for name in TARGETS:
        w = _load_tensor(name, device=device).to(torch.float32)
        print(f"\n[m1b] {name}  shape={tuple(w.shape)}  |w|max={w.abs().max():.3f}")
        for lmax in (5, 7, 9, 12):
            out.append(_run(name, w, lmax))
        del w
        torch.cuda.empty_cache()
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "m1b_probe_rownorm.json").write_text(json.dumps(out, indent=2))
    print(f"\n[m1b] wrote {results_dir / 'm1b_probe_rownorm.json'}")


if __name__ == "__main__":
    main()
