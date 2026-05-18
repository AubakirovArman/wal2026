"""M8a: FP8 vs BF16 microbench + correctness check on real Llama-70B weights.

Validates whether torch._scaled_mm fp8 rowwise path matches bf16 F.linear
quality on our route-decoded weights, and measures the speed / VRAM ratio
on H200.
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
from dwl2_dynamic_route.src.runtime import (
    PackedIDRouteLinear, EagerBf16Linear, quantize_linear_to_packed,
)

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
FP8_MAX = 448.0  # e4m3fn
torch.manual_seed(0)


def load_w(name, device="cuda:0"):
    with safe_open(MODEL_DIR / INDEX[name], framework="pt", device=device) as f:
        return f.get_tensor(name).to(torch.bfloat16)


def make_eager_bf16(w):
    out_f, in_f = w.shape
    lin = nn.Linear(in_f, out_f, bias=False, device=w.device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w)
    eager, _ = quantize_linear_to_packed(lin, runtime_cls=EagerBf16Linear)
    return eager.to(w.device)


def quantize_weight_to_fp8_rowwise(w_bf16: torch.Tensor):
    """Per-output-row symmetric fp8_e4m3 quantization of weight."""
    w = w_bf16.float()
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # [N,1]
    scale = (amax / FP8_MAX).to(torch.float32)                  # [N,1] -> dequant factor
    w_q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_q.contiguous(), scale


def fp8_linear(x_bf16: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor):
    """x @ W.T using torch._scaled_mm with per-token x scale, per-row w scale."""
    orig = x_bf16.shape
    x = x_bf16.reshape(-1, orig[-1])
    x_amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).float()
    x_scale = (x_amax / FP8_MAX).to(torch.float32)
    x_q = (x.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # w_fp8 is [N,K] row-major. b for scaled_mm needs [K,N] column-major (= W.t() with K-major
    # strides, which is what .t() yields when W is contiguous row-major).
    # scale_b shape: rowwise mode wants [1,N].
    out = torch._scaled_mm(
        x_q, w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_scale.view(1, -1),
        out_dtype=torch.bfloat16,
    )
    return out.reshape(*orig[:-1], w_fp8.shape[0])


def bench_call(fn, x, warmup=3, iters=20):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, y


def study_layer(name, configs):
    print(f"\n=== {name} ===", flush=True)
    w = load_w(name)
    out_f, in_f = w.shape

    # Three reference points
    eager = make_eager_bf16(w)            # route-decoded bf16 (our baseline of correctness)
    w_route_bf16 = eager.weight           # route-decoded bf16 weight
    w_fp8, w_scale = quantize_weight_to_fp8_rowwise(w_route_bf16)

    # Storage sizes
    bf16_mb = w_route_bf16.numel() * 2 / 2**20
    fp8_mb = w_fp8.numel() + w_scale.numel() * 4
    fp8_mb /= 2**20
    print(f"  storage: bf16={bf16_mb:.1f} MB  fp8+scale={fp8_mb:.1f} MB  ratio={fp8_mb/bf16_mb:.2f}", flush=True)

    rows = []
    for bs, seq in configs:
        x = torch.randn(bs, seq, in_f, device=w.device, dtype=torch.bfloat16)

        # 1. Reference: F.linear on route-decoded bf16 weight (our current best)
        t_ref, y_ref = bench_call(lambda x: F.linear(x, w_route_bf16), x)
        # 2. FP8 path
        t_fp8, y_fp8 = bench_call(lambda x: fp8_linear(x, w_fp8, w_scale), x)
        # Quality
        rel = float(((y_ref.float() - y_fp8.float()).square().mean() /
                     y_ref.float().square().mean().clamp_min(1e-12)).item())
        max_abs = float((y_ref.float() - y_fp8.float()).abs().max().item())
        finite = bool(torch.isfinite(y_fp8).all().item())
        row = {
            "tensor": name, "bs": bs, "seq": seq,
            "bf16_ms": round(t_ref*1000, 3),
            "fp8_ms": round(t_fp8*1000, 3),
            "speedup": round(t_ref / t_fp8, 2),
            "rel_mse": rel,
            "max_abs_diff": max_abs,
            "finite": finite,
            "ref_amax": float(y_ref.float().abs().max().item()),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    del eager, w_fp8, w_scale, w_route_bf16, w
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    configs = [(1, 1), (1, 32), (1, 512), (1, 2048), (4, 2048)]
    tensors = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]
    all_rows = []
    for t in tensors:
        all_rows.extend(study_layer(t, configs))
    p = ROOT / "dwl2_dynamic_route/results/m8a_fp8_microbench.json"
    p.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {p}")
