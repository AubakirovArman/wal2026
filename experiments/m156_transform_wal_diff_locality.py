"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M156 — Transform-WAL Diff Locality.

Compares diff locality for Raw-WAL vs Transform-WAL under synthetic edits.
Metrics: target diff %, non-target diff %, patch size, RLE/bitmask compression.
Uses CPU for encoding to avoid GPU OOM.
"""
import torch
import time
import json
import math
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def hadamard_matrix(n, device='cpu'):
    if n == 1:
        return torch.ones(1, 1, device=device)
    H = hadamard_matrix(n // 2, device=device)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


def pad_to_power_of_2(x, dim):
    size = x.shape[dim]
    if size <= 1:
        return x, size
    target = 2 ** math.ceil(math.log2(size))
    if target == size:
        return x, size
    pad = target - size
    pads = [0, 0] * x.ndim
    pads[(x.ndim - dim) * 2 - 1] = pad
    return torch.nn.functional.pad(x, pads), size


def apply_hadamard_ortho(W):
    device = W.device
    out_d, in_d = W.shape
    W_pad, orig_out = pad_to_power_of_2(W, 0)
    W_pad, orig_in = pad_to_power_of_2(W_pad, 1)
    out_p, in_p = W_pad.shape
    H_out = hadamard_matrix(out_p, device=device) / math.sqrt(out_p)
    H_in = hadamard_matrix(in_p, device=device) / math.sqrt(in_p)
    W_t = H_out @ W_pad @ H_in.T
    return W_t, (H_out, orig_out, out_p), (H_in, orig_in, in_p)


def inverse_hadamard_ortho(W_t, meta_out, meta_in):
    H_out, orig_out, out_p = meta_out
    H_in, orig_in, in_p = meta_in
    W_inv = H_out.T @ W_t @ H_in
    return W_inv[:orig_out, :orig_in]


def apply_randorth(W):
    out_d, in_d = W.shape
    Q_out = torch.linalg.qr(torch.randn(out_d, out_d, device=W.device))[0]
    Q_in = torch.linalg.qr(torch.randn(in_d, in_d, device=W.device))[0]
    W_t = Q_out @ W @ Q_in.T
    return W_t, Q_out, Q_in


def inverse_randorth(W_t, Q_out, Q_in):
    return Q_out.T @ W_t @ Q_in


def encode_raw(w, atoms, coeffs):
    flat = w.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog


def encode_hadamard(w, atoms, coeffs):
    W_t, meta_out, meta_in = apply_hadamard_ortho(w)
    flat = W_t.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog, meta_out, meta_in


def encode_randorth(w, atoms, coeffs):
    W_t, Q_out, Q_in = apply_randorth(w)
    flat = W_t.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog, Q_out, Q_in


def compute_diff(prog_base, prog_edit):
    diff_atoms = (prog_base.atom_ids != prog_edit.atom_ids).float().mean().item()
    diff_coeffs = (prog_base.coeff_ids != prog_edit.coeff_ids).float().mean().item()
    diff_both = ((prog_base.atom_ids != prog_edit.atom_ids) | 
                 (prog_base.coeff_ids != prog_edit.coeff_ids)).float().mean().item()
    return diff_atoms, diff_coeffs, diff_both


def compute_patch_size(prog_base, prog_edit):
    n = prog_base.N
    changed = ((prog_base.atom_ids != prog_edit.atom_ids) | 
               (prog_base.coeff_ids != prog_edit.coeff_ids)).cpu().numpy()
    
    # Bitmask
    import numpy as np
    bitmask_bytes = (n + 7) // 8
    n_changed = changed.sum()
    patch_bytes = bitmask_bytes + n_changed * 2  # atom_id + coeff_id
    
    # RLE (simplified)
    runs = 1
    for i in range(1, n):
        if changed[i] != changed[i-1]:
            runs += 1
    rle_bytes = runs * 4  # (count, value) pairs
    
    return {
        'bitmask_bytes': patch_bytes,
        'bitmask_mb': patch_bytes / 1e6,
        'rle_bytes': rle_bytes,
        'rle_mb': rle_bytes / 1e6,
    }


def main():
    print("=" * 60)
    print("M156 — Transform-WAL Diff Locality")
    print("=" * 60)
    
    # Load model
    print("\nLoading Llama-3.1-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Select layers
    layers_to_test = [0, 16, 31]
    modules = ['q_proj', 'v_proj']
    
    K, C = 256, 16
    
    # Build global atom tables: raw, hadamard, randorth
    print("\nBuilding atom tables...")
    all_weights = []
    for layer_idx in layers_to_test:
        layer = model.model.layers[layer_idx]
        for mod_name in modules:
            w = getattr(layer.self_attn, mod_name).weight.data
            all_weights.append(w.cpu().float())
    
    # Raw atoms
    all_flat = torch.cat([w.reshape(-1) for w in all_weights])
    atoms_raw = build_l0_atoms(all_flat, K=K, iters=2)
    coeffs_raw = build_coeff_table(all_flat, atoms_raw, C=C, iters=2)
    atoms_raw = atoms_raw[torch.argsort(atoms_raw.abs())]
    
    # Hadamard atoms (in transform space)
    all_hadamard = torch.cat([apply_hadamard_ortho(w)[0].reshape(-1) for w in all_weights])
    atoms_h = build_l0_atoms(all_hadamard, K=K, iters=2)
    coeffs_h = build_coeff_table(all_hadamard, atoms_h, C=C, iters=2)
    atoms_h = atoms_h[torch.argsort(atoms_h.abs())]
    
    # RandOrth atoms
    all_orth = torch.cat([apply_randorth(w)[0].reshape(-1) for w in all_weights])
    atoms_o = build_l0_atoms(all_orth, K=K, iters=2)
    coeffs_o = build_coeff_table(all_orth, atoms_o, C=C, iters=2)
    atoms_o = atoms_o[torch.argsort(atoms_o.abs())]
    
    results = []
    
    for layer_idx in layers_to_test:
        layer = model.model.layers[layer_idx]
        for mod_name in modules:
            w = getattr(layer.self_attn, mod_name).weight.data.cpu().float()
            path = f"layers.{layer_idx}.self_attn.{mod_name}"
            
            # Synthetic edit: perturb 5% of weights
            mask = torch.rand(w.shape) < 0.05
            w_edit = w.clone()
            w_edit[mask] += torch.randn(mask.sum().item()) * 0.01
            
            # Raw-WAL
            prog_base_raw = encode_raw(w, atoms_raw, coeffs_raw)
            prog_edit_raw = encode_raw(w_edit, atoms_raw, coeffs_raw)
            diff_raw = compute_diff(prog_base_raw, prog_edit_raw)
            patch_raw = compute_patch_size(prog_base_raw, prog_edit_raw)
            
            # Hadamard-WAL
            prog_base_h, meta_out_h, meta_in_h = encode_hadamard(w, atoms_h, coeffs_h)
            prog_edit_h, _, _ = encode_hadamard(w_edit, atoms_h, coeffs_h)
            diff_h = compute_diff(prog_base_h, prog_edit_h)
            patch_h = compute_patch_size(prog_base_h, prog_edit_h)
            
            # RandOrth-WAL
            prog_base_o, Q_out_o, Q_in_o = encode_randorth(w, atoms_o, coeffs_o)
            prog_edit_o, _, _ = encode_randorth(w_edit, atoms_o, coeffs_o)
            diff_o = compute_diff(prog_base_o, prog_edit_o)
            patch_o = compute_patch_size(prog_base_o, prog_edit_o)
            
            print(f"\n{path}")
            print(f"  Raw:      diff={diff_raw[2]:.3f}, patch={patch_raw['bitmask_mb']:.3f} MB")
            print(f"  Hadamard: diff={diff_h[2]:.3f}, patch={patch_h['bitmask_mb']:.3f} MB")
            print(f"  RandOrth: diff={diff_o[2]:.3f}, patch={patch_o['bitmask_mb']:.3f} MB")
            
            results.append({
                'path': path,
                'raw_diff': diff_raw[2],
                'raw_patch_mb': patch_raw['bitmask_mb'],
                'hadamard_diff': diff_h[2],
                'hadamard_patch_mb': patch_h['bitmask_mb'],
                'randorth_diff': diff_o[2],
                'randorth_patch_mb': patch_o['bitmask_mb'],
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_raw = sum(r['raw_diff'] for r in results) / len(results)
    avg_h = sum(r['hadamard_diff'] for r in results) / len(results)
    avg_o = sum(r['randorth_diff'] for r in results) / len(results)
    
    avg_raw_patch = sum(r['raw_patch_mb'] for r in results) / len(results)
    avg_h_patch = sum(r['hadamard_patch_mb'] for r in results) / len(results)
    avg_o_patch = sum(r['randorth_patch_mb'] for r in results) / len(results)
    
    print(f"\nAverage diff:")
    print(f"  Raw-WAL:      {avg_raw:.3f}")
    print(f"  Hadamard-WAL: {avg_h:.3f} ({avg_h/avg_raw:.3f}x)")
    print(f"  RandOrth-WAL: {avg_o:.3f} ({avg_o/avg_raw:.3f}x)")
    
    print(f"\nAverage patch size (bitmask):")
    print(f"  Raw-WAL:      {avg_raw_patch:.3f} MB")
    print(f"  Hadamard-WAL: {avg_h_patch:.3f} MB ({avg_h_patch/avg_raw_patch:.3f}x)")
    print(f"  RandOrth-WAL: {avg_o_patch:.3f} MB ({avg_o_patch/avg_raw_patch:.3f}x)")
    
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m156_transform_wal_diff_locality.json'
    with open(out_path, 'w') as f:
        json.dump({
            'avg_raw_diff': avg_raw,
            'avg_hadamard_diff': avg_h,
            'avg_randorth_diff': avg_o,
            'avg_raw_patch_mb': avg_raw_patch,
            'avg_hadamard_patch_mb': avg_h_patch,
            'avg_randorth_patch_mb': avg_o_patch,
            'per_layer': results,
        }, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
