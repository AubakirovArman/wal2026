"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M157 v2 — Transform Vocabulary Study (fast CPU version).

Compares vocabulary strategies:
B. transform-specific global atoms
C. per-module transform atoms
D. per-transform per-module atoms
"""
import torch
import time
import json
import math
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def hadamard_matrix(n):
    if n == 1: return torch.ones(1, 1)
    H = hadamard_matrix(n // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


def pad_to_power_of_2(x, dim):
    size = x.shape[dim]
    if size <= 1: return x, size
    target = 2 ** math.ceil(math.log2(size))
    if target == size: return x, size
    pad = target - size
    pads = [0, 0] * x.ndim
    pads[(x.ndim - dim) * 2 - 1] = pad
    return torch.nn.functional.pad(x, pads), size


def apply_hadamard(W):
    out_d, in_d = W.shape
    W_pad, oo = pad_to_power_of_2(W, 0)
    W_pad, oi = pad_to_power_of_2(W_pad, 1)
    op, ip = W_pad.shape
    H_out = hadamard_matrix(op) / math.sqrt(op)
    H_in = hadamard_matrix(ip) / math.sqrt(ip)
    return H_out @ W_pad @ H_in.T


def encode(w, atoms, coeffs):
    flat = w.reshape(-1)
    prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    mse = ((flat - recon) ** 2).mean().item()
    return mse, prog


def main():
    print("=" * 60)
    print("M157 v2 — Transform Vocabulary Study")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 16]
    modules = ['q_proj', 'v_proj']
    K, C = 64, 8
    
    # Collect weights
    weights = {}
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            weights[f"{li}_{m}"] = w
    
    # Strategy B: transform-specific global atoms
    print("\n--- Strategy B: transform-specific global ---")
    # Raw
    all_raw = torch.cat([w.reshape(-1) for w in weights.values()])
    atoms_b_raw = build_l0_atoms(all_raw, K=K, iters=1)
    coeffs_b_raw = build_coeff_table(all_raw, atoms_b_raw, C=C, iters=1)
    atoms_b_raw = atoms_b_raw[torch.argsort(atoms_b_raw.abs())]
    
    # Hadamard
    all_h = torch.cat([apply_hadamard(w).reshape(-1) for w in weights.values()])
    atoms_b_h = build_l0_atoms(all_h, K=K, iters=1)
    coeffs_b_h = build_coeff_table(all_h, atoms_b_h, C=C, iters=1)
    atoms_b_h = atoms_b_h[torch.argsort(atoms_b_h.abs())]
    
    # Strategy C: per-module transform atoms
    print("--- Strategy C: per-module transform ---")
    atoms_c = {}
    for key, w in weights.items():
        for transform, w_t in [('raw', w), ('hadamard', apply_hadamard(w))]:
            a = build_l0_atoms(w_t.reshape(-1), K=K, iters=1)
            c = build_coeff_table(w_t.reshape(-1), a, C=C, iters=1)
            atoms_c[f"{key}_{transform}"] = (a[torch.argsort(a.abs())], c)
    
    # Strategy D: per-transform per-module (same as C for this scale)
    # For 2 layers × 2 modules, C and D are identical
    
    # Evaluate
    results = []
    
    for key, w in weights.items():
        w_h = apply_hadamard(w)
        
        # B: raw
        mse_b_raw, _ = encode(w, atoms_b_raw, coeffs_b_raw)
        # B: hadamard
        mse_b_h, _ = encode(w_h, atoms_b_h, coeffs_b_h)
        
        # C: raw
        mse_c_raw, _ = encode(w, *atoms_c[f"{key}_raw"])
        # C: hadamard
        mse_c_h, _ = encode(w_h, *atoms_c[f"{key}_hadamard"])
        
        results.append({
            'key': key,
            'mse_b_raw': mse_b_raw,
            'mse_b_hadamard': mse_b_h,
            'mse_c_raw': mse_c_raw,
            'mse_c_hadamard': mse_c_h,
        })
        
        print(f"  {key}: B_raw={mse_b_raw:.2e} B_h={mse_b_h:.2e} C_raw={mse_c_raw:.2e} C_h={mse_c_h:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for strategy, raw_key, h_key in [
        ('B (global)', 'mse_b_raw', 'mse_b_hadamard'),
        ('C (per-module)', 'mse_c_raw', 'mse_c_hadamard'),
    ]:
        avg_raw = sum(r[raw_key] for r in results) / len(results)
        avg_h = sum(r[h_key] for r in results) / len(results)
        print(f"  {strategy}: raw={avg_raw:.2e}, hadamard={avg_h:.2e}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m157_transform_vocab_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
