"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""
M147 / Track 10: WAL-Friendly Training Probe

Goal: Test if WAL-aware regularization improves weight compatibility with quantization.

Method:
  1. Build atom table from model
  2. For each weight, find nearest atom×coeff
  3. Measure "WAL distance" = mean |weight - recon|
  4. Test simple regularizer: push weights toward nearest atom×coeff
  5. Measure improvement after one gradient step
  6. Compare with L2 regularization baseline
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

DEVICE = "cuda:0"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
SEED = 42
K, C = 256, 16

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    return atoms[perm], perm


def build_table_from_model(model, K=256, C=16, seed=42):
    torch.manual_seed(seed)
    all_weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name and p.ndim >= 2:
            all_weights.append(p.data.float().cpu().reshape(-1))
    pooled = torch.cat(all_weights)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=2)
    sorted_atoms, perm = canonicalize_atoms(atoms)
    return sorted_atoms, coeffs


def compute_wal_distance(weight, atoms, coeffs):
    """Compute mean distance from each weight to nearest atom×coeff."""
    w = weight.float().flatten()
    atoms_f = atoms.to(torch.float32)
    coeffs_f = coeffs.to(torch.float32)
    N, M = len(atoms_f), len(coeffs_f)
    
    # Find best atom×coeff for each weight (chunked)
    chunk_size = 50000
    total_dist = 0
    total_count = 0
    
    for i_start in range(0, len(w), chunk_size):
        i_end = min(i_start + chunk_size, len(w))
        w_chunk = w[i_start:i_end]
        recon_grid = atoms_f.unsqueeze(0).unsqueeze(2) * coeffs_f.unsqueeze(0).unsqueeze(1)
        diffs = (recon_grid - w_chunk.unsqueeze(1).unsqueeze(2)).abs()
        best_err = diffs.view(len(w_chunk), -1).min(dim=1).values
        total_dist += best_err.sum().item()
        total_count += len(w_chunk)
    
    return total_dist / total_count


def apply_wal_regularizer(weight, atoms, coeffs, lr=0.01):
    """One gradient step toward nearest atom×coeff."""
    w = weight.float().clone().flatten()
    atoms_f = atoms.to(torch.float32)
    coeffs_f = coeffs.to(torch.float32)
    N, M = len(atoms_f), len(coeffs_f)
    
    # Find nearest recon for each weight
    chunk_size = 50000
    recon = torch.zeros_like(w)
    
    for i_start in range(0, len(w), chunk_size):
        i_end = min(i_start + chunk_size, len(w))
        w_chunk = w[i_start:i_end]
        # recon_grid: [chunk, N, M]
        recon_grid = atoms_f.unsqueeze(0).unsqueeze(2) * coeffs_f.unsqueeze(0).unsqueeze(1)
        # w_chunk: [chunk] → [chunk, 1, 1]
        diffs = (recon_grid - w_chunk.unsqueeze(1).unsqueeze(2)).abs()
        best_idx = diffs.view(len(w_chunk), -1).argmin(dim=1)
        best_atoms = best_idx // M
        best_coeffs = best_idx % M
        recon[i_start:i_end] = atoms_f[best_atoms] * coeffs_f[best_coeffs]
    
    # Gradient step toward recon
    with torch.no_grad():
        w_new = w - lr * (w - recon)
    
    return w_new.reshape(weight.shape)


def test_layer(layer_name, layer, atoms, coeffs):
    """Test WAL distance before and after regularizer."""
    w = layer.weight.data
    
    # Before
    dist_before = compute_wal_distance(w, atoms, coeffs)
    
    # After WAL regularizer
    w_reg = apply_wal_regularizer(w, atoms, coeffs, lr=0.01)
    dist_after = compute_wal_distance(w_reg, atoms, coeffs)
    
    # After L2 regularizer (baseline)
    w_l2 = w * 0.99  # Simple shrinkage
    dist_l2 = compute_wal_distance(w_l2, atoms, coeffs)
    
    return {
        'dist_before': dist_before,
        'dist_after': dist_after,
        'dist_l2': dist_l2,
        'improvement': (dist_before - dist_after) / dist_before * 100,
        'l2_improvement': (dist_before - dist_l2) / dist_before * 100,
    }


def main():
    print("=" * 70)
    print("M147 / Track 10: WAL-Friendly Training Probe")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Build table
    print("[2] Building atom table...")
    atoms, coeffs = build_table_from_model(model, K=K, C=C, seed=SEED)
    print(f"  Atoms: {atoms[:5].tolist()}")
    print(f"  Coeffs: {coeffs[:5].tolist()}")

    # 3. Test on key layers
    layer_names = [
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.15.self_attn.v_proj',
        'model.layers.15.mlp.gate_proj',
        'model.layers.30.self_attn.o_proj',
    ]
    
    print(f"[3] Testing WAL regularizer on {len(layer_names)} layers...")
    results = []
    
    for name in layer_names:
        parts = name.split('.')
        layer = model
        for p in parts:
            layer = getattr(layer, p)
        
        result = test_layer(name, layer, atoms, coeffs)
        results.append({
            'layer': name,
            **result,
        })
        print(f"  {name:50s}")
        print(f"    Before: {result['dist_before']:.6f}")
        print(f"    After WAL reg: {result['dist_after']:.6f} ({result['improvement']:+.1f}%)")
        print(f"    After L2 shrink: {result['dist_l2']:.6f} ({result['l2_improvement']:+.1f}%)")

    # 4. Summary
    avg_before = sum(r['dist_before'] for r in results) / len(results)
    avg_after = sum(r['dist_after'] for r in results) / len(results)
    avg_l2 = sum(r['dist_l2'] for r in results) / len(results)
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    avg_l2_improvement = sum(r['l2_improvement'] for r in results) / len(results)
    
    print(f"\n[4] Average across all layers:")
    print(f"  Before:        {avg_before:.6f}")
    print(f"  WAL reg:       {avg_after:.6f} ({avg_improvement:+.1f}%)")
    print(f"  L2 shrink:     {avg_l2:.6f} ({avg_l2_improvement:+.1f}%)")
    
    if avg_improvement > avg_l2_improvement + 5:
        print(f"  ✅ WAL regularizer outperforms L2 by {avg_improvement - avg_l2_improvement:.1f}%")
    elif avg_improvement > 0:
        print(f"  ⚠️  WAL regularizer helps but not much ({avg_improvement:.1f}%)")
    else:
        print(f"  ❌ WAL regularizer hurts ({avg_improvement:.1f}%)")

    # 5. Save
    output = {
        'layers': results,
        'average': {
            'dist_before': avg_before,
            'dist_after': avg_after,
            'dist_l2': avg_l2,
            'improvement': avg_improvement,
            'l2_improvement': avg_l2_improvement,
        },
    }
    
    out_path = 'experiments/m147_wal_friendly_training.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M147 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
