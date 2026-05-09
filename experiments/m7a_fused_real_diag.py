"""Fast real-weight fused kernel diag — loads single weights via safetensors.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dwl2_dynamic_route.src.runtime import (
    PackedIDRouteLinear, FusedIDRouteLinear, quantize_linear_to_packed,
)

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]

torch.manual_seed(0)


def load_weight(tensor_name: str, device="cuda:0") -> torch.Tensor:
    shard = MODEL_DIR / INDEX[tensor_name]
    with safe_open(shard, framework="pt", device=device) as f:
        return f.get_tensor(tensor_name).to(torch.bfloat16)


def probe(tensor_name: str, seq_lens=(32, 512, 2048), device="cuda:0"):
    w = load_weight(tensor_name, device)
    out_f, in_f = w.shape
    linear = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(w)

    print(f"W absmax={w.float().abs().max().item():.4f} absmean={w.float().abs().mean().item():.4f} shape={tuple(w.shape)}", flush=True)

    ref_packed, stats = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    ref_packed = ref_packed.to(device)
    fused_packed, _ = quantize_linear_to_packed(linear, runtime_cls=FusedIDRouteLinear)
    fused_packed = fused_packed.to(device)

    print(f"  codebook_sum: dtype={ref_packed.codebook_sum.dtype} absmax={ref_packed.codebook_sum.float().abs().max().item():.4f} "
          f"M={ref_packed.codebook_sum.numel()}")
    print(f"  row_scale: absmax={ref_packed.row_scale.float().abs().max().item():.4f}  rel_mse={stats['rel_mse']:.3e}")

    res = []
    for seq in seq_lens:
        x = torch.randn(1, seq, in_f, device=device, dtype=torch.bfloat16)
        y_ref = ref_packed(x)
        y_fused = fused_packed(x)
        fr, ff = bool(torch.isfinite(y_ref).all()), bool(torch.isfinite(y_fused).all())
        if fr and ff:
            rel = float(((y_ref.float() - y_fused.float()).square().mean() /
                         y_ref.float().square().mean().clamp_min(1e-12)).item())
        else:
            rel = float('nan')
        print(f"  seq={seq:4d}  finite_ref={fr} finite_fused={ff} rel_mse={rel:.3e}  "
              f"ref_max={y_ref.float().abs().max().item():.2f} fused_max={y_fused.float().abs().max().item():.2f}", flush=True)
        res.append({"seq": seq, "finite_ref": fr, "finite_fused": ff, "rel_mse": rel,
                    "ref_abs_max": float(y_ref.float().abs().max().item()),
                    "fused_abs_max": float(y_fused.float().abs().max().item())})

    # Real activations test: use actual hidden state from a pre-rms-norm approximate range
    # Llama hidden states post-RMSNorm are roughly N(0, 1); after attn/mlp can reach large values.
    # Stress with larger input scale:
    for scale in (3.0, 10.0):
        x = torch.randn(1, 2048, in_f, device=device, dtype=torch.bfloat16) * scale
        y_ref = ref_packed(x); y_fused = fused_packed(x)
        fr, ff = bool(torch.isfinite(y_ref).all()), bool(torch.isfinite(y_fused).all())
        if fr and ff:
            rel = float(((y_ref.float() - y_fused.float()).square().mean() /
                         y_ref.float().square().mean().clamp_min(1e-12)).item())
        else:
            rel = float('nan')
        print(f"  scale={scale}  finite_ref={fr} finite_fused={ff} rel_mse={rel:.3e} "
              f"ref_max={y_ref.float().abs().max().item():.1f} fused_max={y_fused.float().abs().max().item():.1f}",
              flush=True)
        res.append({"scale": scale, "finite_ref": fr, "finite_fused": ff, "rel_mse": rel,
                    "ref_abs_max": float(y_ref.float().abs().max().item()),
                    "fused_abs_max": float(y_fused.float().abs().max().item())})
    del ref_packed, fused_packed, linear, w
    torch.cuda.empty_cache()
    return {"tensor": tensor_name, "stats": stats, "probes": res}


if __name__ == "__main__":
    out = []
    for t in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.40.mlp.down_proj.weight",
    ]:
        print(f"\n=== {t} ===", flush=True)
        out.append(probe(t))
    p = ROOT / "dwl2_dynamic_route/results/m7a_fused_real_diag.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
