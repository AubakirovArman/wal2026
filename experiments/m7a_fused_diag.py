"""Root-cause diagnostic for FusedIDRouteLinear NaN.

Quantize a single Llama linear, compare fused vs reference on synthetic input,
and dump where they diverge in magnitude/finiteness.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.runtime import (
    PackedIDRouteLinear,
    FusedIDRouteLinear,
    quantize_linear_to_packed,
)

torch.manual_seed(0)


def diag_one(out_features: int, in_features: int, device: str = "cuda:0") -> dict:
    linear = nn.Linear(in_features, out_features, bias=False).to(device=device, dtype=torch.bfloat16)
    # Simulate llama-ish distribution: most small, a few outliers
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        linear.weight[torch.rand_like(linear.weight) < 1e-4] *= 20.0

    # Reference (materialize)
    ref_packed, stats = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    ref_packed = ref_packed.to(device)

    # Fused
    fused_packed, _ = quantize_linear_to_packed(linear, runtime_cls=FusedIDRouteLinear)
    fused_packed = fused_packed.to(device)

    x = torch.randn(32, in_features, device=device, dtype=torch.bfloat16)
    y_ref = ref_packed(x)
    y_fused = fused_packed(x)

    finite_ref = bool(torch.isfinite(y_ref).all().item())
    finite_fused = bool(torch.isfinite(y_fused).all().item())
    rel = float(((y_ref.float() - y_fused.float()).square().mean() /
                 (y_ref.float().square().mean().clamp_min(1e-12))).item())
    max_abs_diff = float((y_ref.float() - y_fused.float()).abs().max().item())
    return {
        "shape": (out_features, in_features),
        "finite_ref": finite_ref,
        "finite_fused": finite_fused,
        "rel_mse": rel,
        "max_abs_diff": max_abs_diff,
        "ref_abs_max": float(y_ref.float().abs().max().item()),
        "fused_abs_max": float(y_fused.float().abs().max().item()),
        "codebook_sum_dtype": str(ref_packed.codebook_sum.dtype),
        "codebook_sum_absmax": float(ref_packed.codebook_sum.float().abs().max().item()),
        "row_scale_absmax": float(ref_packed.row_scale.float().abs().max().item()),
        "codebook_size": int(ref_packed.codebook_sum.numel()),
        **stats,
    }


if __name__ == "__main__":
    import json

    results = []
    for shape in [(4096, 4096), (8192, 8192), (28672, 8192)]:
        print(f"=== {shape} ===", flush=True)
        r = diag_one(*shape)
        print(json.dumps(r, indent=2))
        results.append(r)
    out = ROOT / "dwl2_dynamic_route/results/m7a_fused_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
