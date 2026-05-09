"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""
M141 / Track 4: Re-Encode Geometry / Safety Score (FAST VERSION)

Goal: Predict which edits survive re-encode without full PPL evaluation.
Build correlation between edit geometry metrics and re-encode residual.

Uses dense model + synthetic edits + simple quantization residual as proxy for PPL loss.
"""

import os, sys, json, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEVICE = "cuda:0"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
TARGET_LAYER = 'model.layers.15.self_attn.q_proj'

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def simple_quantize_residual(weight, num_bins=256):
    """Simple uniform quantization + residual as proxy for re-encode loss."""
    w = weight.float()
    min_val, max_val = w.min(), w.max()
    scale = (max_val - min_val) / (num_bins - 1)
    quantized = ((w - min_val) / scale).round().clamp(0, num_bins - 1)
    recon = quantized * scale + min_val
    residual = (w - recon).abs().mean().item()
    return residual


def compute_geometry_metrics(base_weight, edited_weight):
    """Compute geometric metrics for an edit."""
    delta = (edited_weight - base_weight).float()
    
    frobenius = torch.norm(delta, p='fro').item()
    spectral = torch.linalg.matrix_norm(delta, ord=2).item()
    max_abs = delta.abs().max().item()
    mean_abs = delta.abs().mean().item()
    std_abs = delta.abs().std().item()
    
    # Kurtosis (heavy-tailness)
    delta_flat = delta.flatten()
    mean_d = delta_flat.mean()
    std_d = delta_flat.std()
    if std_d > 1e-6:
        kurtosis = ((delta_flat - mean_d) ** 4).mean() / (std_d ** 4)
    else:
        kurtosis = 0.0
    
    return {
        'frobenius': frobenius,
        'spectral': spectral,
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'std_abs': std_abs,
        'kurtosis': kurtosis.item() if torch.is_tensor(kurtosis) else kurtosis,
    }


def get_layer(model, name):
    parts = name.split('.')
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    return layer


def main():
    print("=" * 70)
    print("M141 / Track 4: Re-Encode Geometry / Safety Score")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Get target layer
    layer = get_layer(model, TARGET_LAYER)
    base_weight = layer.weight.data.clone()
    
    # 3. Base quantization residual
    base_residual = simple_quantize_residual(base_weight)
    print(f"[2] Base quantization residual: {base_residual:.6f}")

    # 4. Test different edit magnitudes
    magnitudes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    results = []

    print(f"[3] Testing {len(magnitudes)} edit magnitudes...")
    for i, mag in enumerate(magnitudes):
        # Apply synthetic edit
        torch.manual_seed(42 + i)
        edit = torch.randn_like(base_weight) * mag
        edited = base_weight + edit
        
        # Geometry metrics
        metrics = compute_geometry_metrics(base_weight, edited)
        
        # Quantization residual (proxy for re-encode loss)
        residual = simple_quantize_residual(edited)
        delta_residual = residual - base_residual
        
        print(f"\n  mag={mag:.4f}: Frobenius={metrics['frobenius']:.4f}, Spectral={metrics['spectral']:.4f}, Max={metrics['max_abs']:.6f}, Mean={metrics['mean_abs']:.6f}")
        print(f"           Quant residual: {residual:.6f} (Δ{delta_residual:+.6f}), Kurtosis={metrics['kurtosis']:.2f}")
        
        results.append({
            'magnitude': mag,
            'quant_residual': residual,
            'delta_residual': delta_residual,
            **metrics,
        })

    # 5. Compute correlations
    print("\n[4] Computing correlations with quantization residual...")
    import numpy as np
    
    keys = ['frobenius', 'spectral', 'max_abs', 'mean_abs', 'std_abs', 'kurtosis']
    
    correlations = {}
    for key in keys:
        x = np.array([r[key] for r in results])
        y = np.array([r['delta_residual'] for r in results])
        if x.std() > 0 and y.std() > 0:
            corr = np.corrcoef(x, y)[0, 1]
            correlations[key] = float(corr)
        else:
            correlations[key] = 0.0
    
    print(f"\n  Correlations with ΔQuantResidual:")
    for key, corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        print(f"    {key:20s}: {corr:+.4f}")

    # 6. Propose Safety Score
    top_3 = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:3]
    print(f"\n  Top 3 predictors: {[k for k, _ in top_3]}")
    
    # Simple weighted score
    def safety_score(frobenius, spectral, max_abs, mean_abs, std_abs, kurtosis):
        # Normalize by max values from our sweep
        max_f = max(r['frobenius'] for r in results)
        max_s = max(r['spectral'] for r in results)
        max_m = max(r['max_abs'] for r in results)
        max_mean = max(r['mean_abs'] for r in results)
        
        # Weighted combination (weights from correlation strength)
        score = 0
        if max_f > 0:
            score += abs(correlations.get('frobenius', 0)) * (frobenius / max_f)
        if max_s > 0:
            score += abs(correlations.get('spectral', 0)) * (spectral / max_s)
        if max_m > 0:
            score += abs(correlations.get('max_abs', 0)) * (max_abs / max_m)
        return score

    # 7. Save results
    output = {
        'target_layer': TARGET_LAYER,
        'base_quant_residual': base_residual,
        'results': results,
        'correlations': correlations,
        'top_predictors': [k for k, _ in top_3],
        'safety_score_formula': 'weighted_norm_of_top_correlated_metrics',
    }
    
    out_path = 'experiments/m141_reencode_geometry.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M141 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
