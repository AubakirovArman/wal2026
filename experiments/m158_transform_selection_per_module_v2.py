#!/usr/bin/env python3
"""M158 v2 — Transform Selection per Module (fast CPU).

Compares: single transform for all modules vs module-specific transforms.
"""
import torch
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
    return ((flat - recon) ** 2).mean().item()


def main():
    print("=" * 60)
    print("M158 v2 — Transform Selection per Module")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 16]
    modules = ['q_proj', 'v_proj', 'gate_proj']
    K, C = 64, 8
    
    # Collect weights
    weights = {}
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn if m != 'gate_proj' else model.model.layers[li].mlp, m).weight.data.float()
            weights[f"{li}_{m}"] = w
    
    # Strategy 1: Single Hadamard for all
    print("\n--- Single Hadamard (all modules) ---")
    all_h = torch.cat([apply_hadamard(w).reshape(-1) for w in weights.values()])
    atoms_single = build_l0_atoms(all_h, K=K, iters=1)
    coeffs_single = build_coeff_table(all_h, atoms_single, C=C, iters=1)
    atoms_single = atoms_single[torch.argsort(atoms_single.abs())]
    
    # Strategy 2: Module-specific Hadamard
    print("--- Module-specific Hadamard ---")
    atoms_mod = {}
    for key, w in weights.items():
        w_h = apply_hadamard(w)
        a = build_l0_atoms(w_h.reshape(-1), K=K, iters=1)
        c = build_coeff_table(w_h.reshape(-1), a, C=C, iters=1)
        atoms_mod[key] = (a[torch.argsort(a.abs())], c)
    
    # Evaluate
    results = []
    for key, w in weights.items():
        w_h = apply_hadamard(w)
        
        mse_single = encode(w_h, atoms_single, coeffs_single)
        mse_specific = encode(w_h, *atoms_mod[key])
        
        results.append({
            'key': key,
            'mse_single': mse_single,
            'mse_specific': mse_specific,
            'ratio': mse_single / max(mse_specific, 1e-10),
        })
        
        print(f"  {key}: single={mse_single:.2e}, specific={mse_specific:.2e}, ratio={mse_single/max(mse_specific,1e-10):.2f}")
    
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    print(f"\nAvg single/specific ratio: {avg_ratio:.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m158_transform_selection_per_module.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
