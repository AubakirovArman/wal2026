#!/usr/bin/env python3
"""M156 v2 — Transform-WAL Diff Locality (fast CPU version).

Compares diff locality for Raw-WAL vs Transform-WAL under synthetic edits.
"""
import torch
import time
import json
import math
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def apply_randorth(W):
    out_d, in_d = W.shape
    Q_out = torch.linalg.qr(torch.randn(out_d, out_d))[0]
    Q_in = torch.linalg.qr(torch.randn(in_d, in_d))[0]
    return Q_out @ W @ Q_in.T, Q_out, Q_in


def inverse_randorth(W_t, Q_out, Q_in):
    return Q_out.T @ W_t @ Q_in


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
    return H_out @ W_pad @ H_in.T, (H_out, oo, op), (H_in, oi, ip)


def inverse_hadamard(W_t, m_out, m_in):
    H_out, oo, op = m_out
    H_in, oi, ip = m_in
    return (H_out.T @ W_t @ H_in)[:oo, :oi]


def encode_raw(w, atoms, coeffs):
    flat = w.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog


def encode_hadamard(w, atoms, coeffs):
    W_t, meta_out, meta_in = apply_hadamard(w)
    flat = W_t.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog, meta_out, meta_in


def encode_randorth(w, atoms, coeffs):
    W_t, Q_out, Q_in = apply_randorth(w)
    flat = W_t.reshape(-1)
    prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return prog, Q_out, Q_in


def compute_diff(prog_base, prog_edit):
    both = ((prog_base.atom_ids != prog_edit.atom_ids) | 
            (prog_base.coeff_ids != prog_edit.coeff_ids)).float().mean().item()
    return both


def compute_patch_size(prog_base, prog_edit):
    n = prog_base.N
    changed = ((prog_base.atom_ids != prog_edit.atom_ids) | 
               (prog_base.coeff_ids != prog_edit.coeff_ids)).cpu().numpy()
    import numpy as np
    bitmask_bytes = (n + 7) // 8 + changed.sum() * 2
    return bitmask_bytes / 1e6


def main():
    print("=" * 60)
    print("M156 v2 — Transform-WAL Diff Locality")
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
    
    # Build atoms
    print("\nBuilding atom tables...")
    weights = []
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            weights.append(w)
    
    # Raw
    all_flat = torch.cat([w.reshape(-1) for w in weights])
    atoms_raw = build_l0_atoms(all_flat, K=K, iters=1)
    coeffs_raw = build_coeff_table(all_flat, atoms_raw, C=C, iters=1)
    atoms_raw = atoms_raw[torch.argsort(atoms_raw.abs())]
    
    # Hadamard
    all_h = torch.cat([apply_hadamard(w)[0].reshape(-1) for w in weights])
    atoms_h = build_l0_atoms(all_h, K=K, iters=1)
    coeffs_h = build_coeff_table(all_h, atoms_h, C=C, iters=1)
    atoms_h = atoms_h[torch.argsort(atoms_h.abs())]
    
    # RandOrth
    all_o = torch.cat([apply_randorth(w)[0].reshape(-1) for w in weights])
    atoms_o = build_l0_atoms(all_o, K=K, iters=1)
    coeffs_o = build_coeff_table(all_o, atoms_o, C=C, iters=1)
    atoms_o = atoms_o[torch.argsort(atoms_o.abs())]
    
    results = []
    
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            path = f"layers.{li}.self_attn.{m}"
            
            w_edit = w + torch.randn_like(w) * 0.001
            
            # Raw
            pb = encode_raw(w, atoms_raw, coeffs_raw)
            pe = encode_raw(w_edit, atoms_raw, coeffs_raw)
            diff_r = compute_diff(pb, pe)
            patch_r = compute_patch_size(pb, pe)
            
            # Hadamard
            pb_h, mh_out, mh_in = encode_hadamard(w, atoms_h, coeffs_h)
            pe_h, _, _ = encode_hadamard(w_edit, atoms_h, coeffs_h)
            diff_h = compute_diff(pb_h, pe_h)
            patch_h = compute_patch_size(pb_h, pe_h)
            
            # RandOrth
            pb_o, Qo, Qi = encode_randorth(w, atoms_o, coeffs_o)
            pe_o, _, _ = encode_randorth(w_edit, atoms_o, coeffs_o)
            diff_o = compute_diff(pb_o, pe_o)
            patch_o = compute_patch_size(pb_o, pe_o)
            
            print(f"\n{path}")
            print(f"  Raw:      diff={diff_r:.3f}, patch={patch_r:.3f} MB")
            print(f"  Hadamard: diff={diff_h:.3f}, patch={patch_h:.3f} MB")
            print(f"  RandOrth: diff={diff_o:.3f}, patch={patch_o:.3f} MB")
            
            results.append({
                'path': path,
                'raw_diff': diff_r, 'raw_patch_mb': patch_r,
                'hadamard_diff': diff_h, 'hadamard_patch_mb': patch_h,
                'randorth_diff': diff_o, 'randorth_patch_mb': patch_o,
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, diff_key, patch_key in [
        ('Raw', 'raw_diff', 'raw_patch_mb'),
        ('Hadamard', 'hadamard_diff', 'hadamard_patch_mb'),
        ('RandOrth', 'randorth_diff', 'randorth_patch_mb'),
    ]:
        avg_diff = sum(r[diff_key] for r in results) / len(results)
        avg_patch = sum(r[patch_key] for r in results) / len(results)
        print(f"  {name:12s}: avg diff={avg_diff:.3f}, avg patch={avg_patch:.3f} MB")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m156_transform_wal_diff_locality.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
