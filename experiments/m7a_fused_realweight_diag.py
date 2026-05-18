"""Real-weight diagnostic for FusedIDRouteLinear NaN on Llama-3.3-70B.

Load one real linear, quantize, compare fused vs materialize on realistic seq_len.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dwl2_dynamic_route.src.runtime import (
    PackedIDRouteLinear, FusedIDRouteLinear, quantize_linear_to_packed,
)

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"

torch.manual_seed(0)


def probe_real(layer_path: str, seq_lens=(32, 512, 2048), device="cuda:3"):
    # Load only that linear from state dict via small model load — use index file
    # Shortcut: load on CPU via low_cpu_mem_usage=True, grab the single linear, move to GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}, local_files_only=True, low_cpu_mem_usage=True,
    )
    mod = model
    for part in layer_path.split("."):
        mod = getattr(mod, part)
    linear = mod.to(device)
    del model
    torch.cuda.empty_cache()

    print(f"weight abs max: {linear.weight.float().abs().max().item():.4f}", flush=True)
    print(f"weight abs mean: {linear.weight.float().abs().mean().item():.4f}", flush=True)

    ref_packed, stats = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    ref_packed = ref_packed.to(device)
    fused_packed, _ = quantize_linear_to_packed(linear, runtime_cls=FusedIDRouteLinear)
    fused_packed = fused_packed.to(device)

    print(f"codebook_sum absmax fp16: {ref_packed.codebook_sum.float().abs().max().item():.4f}", flush=True)
    print(f"row_scale absmax fp16: {ref_packed.row_scale.float().abs().max().item():.4f}", flush=True)
    print(f"codebook size: {ref_packed.codebook_sum.numel()}", flush=True)

    results = []
    in_features = linear.in_features
    for seq_len in seq_lens:
        x = torch.randn(1, seq_len, in_features, device=device, dtype=torch.bfloat16)
        y_ref = ref_packed(x)
        y_fused = fused_packed(x)
        finite_ref = bool(torch.isfinite(y_ref).all().item())
        finite_fused = bool(torch.isfinite(y_fused).all().item())
        if finite_ref and finite_fused:
            rel = float(((y_ref.float() - y_fused.float()).square().mean() /
                         y_ref.float().square().mean().clamp_min(1e-12)).item())
        else:
            rel = float('nan')
        print(f"seq={seq_len:4d} finite_ref={finite_ref} finite_fused={finite_fused} rel_mse={rel:.3e} "
              f"ref_max={y_ref.float().abs().max().item():.3f} fused_max={y_fused.float().abs().max().item():.3f}",
              flush=True)
        results.append({
            "seq": seq_len, "finite_ref": finite_ref, "finite_fused": finite_fused,
            "rel_mse": rel,
            "ref_abs_max": float(y_ref.float().abs().max().item()),
            "fused_abs_max": float(y_fused.float().abs().max().item()),
        })
    return {"layer": layer_path, "stats": stats, "probes": results}


if __name__ == "__main__":
    out = []
    for lp in ["model.language_model.layers.0.self_attn.q_proj",
               "model.language_model.layers.0.mlp.up_proj",
               "model.language_model.layers.40.mlp.down_proj"]:
        print(f"\n=== {lp} ===", flush=True)
        out.append(probe_real(lp))
    p = ROOT / "dwl2_dynamic_route/results/m7a_fused_realweight_diag.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
