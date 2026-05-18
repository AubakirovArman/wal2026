"""M8a-v2: optimize the FP8 path.

Strategies tested:
  - bf16-direct quantize (no fp32 cast)
  - use_fast_accum=True
  - per-tensor x scale (skip per-token amax)
  - precomputed weight scale layout
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
FP8_MAX = 448.0
torch.manual_seed(0)


def load_w(name, device="cuda:0"):
    with safe_open(MODEL_DIR / INDEX[name], framework="pt", device=device) as f:
        return f.get_tensor(name).to(torch.bfloat16)


def quantize_w_fp8(w_bf16):
    amax = w_bf16.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / FP8_MAX).to(torch.float32)
    w_q = (w_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_q.contiguous(), scale


# Variant A: per-token x quantize, fp32 cast (the original)
def fp8_v0(x, w_fp8, w_scale):
    orig = x.shape
    x = x.reshape(-1, orig[-1])
    x_amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).float()
    x_scale = (x_amax / FP8_MAX).to(torch.float32)
    x_q = (x.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x_q, w_fp8.t(), scale_a=x_scale, scale_b=w_scale.view(1, -1),
                           out_dtype=torch.bfloat16)
    return out.reshape(*orig[:-1], w_fp8.shape[0])


# Variant B: bf16-direct quantize (skip fp32 cast)
def fp8_v1(x, w_fp8, w_scale):
    orig = x.shape
    x = x.reshape(-1, orig[-1])
    x_amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    x_scale = (x_amax / FP8_MAX).float()
    x_q = (x / x_amax * FP8_MAX).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x_q, w_fp8.t(), scale_a=x_scale, scale_b=w_scale.view(1, -1),
                           out_dtype=torch.bfloat16, use_fast_accum=True)
    return out.reshape(*orig[:-1], w_fp8.shape[0])


# Variant C: per-tensor x scale (single scalar) - fastest quantize
def fp8_v2(x, w_fp8, w_scale):
    orig = x.shape
    x = x.reshape(-1, orig[-1])
    x_amax = x.abs().amax().clamp_min(1e-8).float()
    x_scale = (x_amax / FP8_MAX).reshape(1, 1)
    x_q = (x / x_amax * FP8_MAX).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # When scale_a is per-tensor, scaled_mm wants [1,1] for both
    out = torch._scaled_mm(x_q, w_fp8.t(), scale_a=x_scale, scale_b=w_scale.view(1, -1),
                           out_dtype=torch.bfloat16, use_fast_accum=True)
    return out.reshape(*orig[:-1], w_fp8.shape[0])


def bench(fn, x, warmup=5, iters=30):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, y


def study(name, configs):
    print(f"\n=== {name} ===", flush=True)
    w = load_w(name)
    out_f, in_f = w.shape
    w_fp8, w_scale = quantize_w_fp8(w)
    print(f"  bf16={w.numel()*2/2**20:.1f} MB  fp8={w_fp8.numel()/2**20 + w_scale.numel()*4/2**20:.1f} MB", flush=True)

    rows = []
    for bs, seq in configs:
        x = torch.randn(bs, seq, in_f, device=w.device, dtype=torch.bfloat16)
        t_bf, y_bf = bench(lambda x: F.linear(x, w), x)
        t_v0, y_v0 = bench(lambda x: fp8_v0(x, w_fp8, w_scale), x)
        t_v1, y_v1 = bench(lambda x: fp8_v1(x, w_fp8, w_scale), x)
        t_v2, y_v2 = bench(lambda x: fp8_v2(x, w_fp8, w_scale), x)
        def relmse(yref, y):
            return float(((yref.float() - y.float()).square().mean() /
                          yref.float().square().mean().clamp_min(1e-12)).item())
        row = {
            "bs": bs, "seq": seq,
            "bf16_ms": round(t_bf*1000, 3),
            "v0_ms": round(t_v0*1000, 3), "v0_x": round(t_bf/t_v0, 2), "v0_rmse": relmse(y_bf, y_v0),
            "v1_ms": round(t_v1*1000, 3), "v1_x": round(t_bf/t_v1, 2), "v1_rmse": relmse(y_bf, y_v1),
            "v2_ms": round(t_v2*1000, 3), "v2_x": round(t_bf/t_v2, 2), "v2_rmse": relmse(y_bf, y_v2),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    del w, w_fp8, w_scale
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import json
        print(f"M8a_v2: FP8 v2 experimental - {e}")
        with open("results/m8a_fp8_v2_microbench.json", "w") as f:
            json.dump({"status": "HARDWARE_LIMITED", "note": f"H200 SM90 FP8 blockwise: {e}"}, f)
    configs = [(1, 1), (1, 32), (1, 512), (1, 2048), (4, 2048), (8, 2048)]
    tensors = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]
    out = []
    for t in tensors:
        out.append({"tensor": t, "rows": study(t, configs)})
    p = ROOT / "dwl2_dynamic_route/results/m8a_fp8_v2_microbench.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
