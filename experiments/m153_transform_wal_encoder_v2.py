"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M153 v2 — Transform-WAL Encoder (fast version).

CPU model, 2 layers, K=64, iters=1, 3 transforms.
"""
import torch
import time
import json
import math
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1, wal_decode_v1


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


def encode(w, transform, atoms, coeffs):
    if transform == 'Raw':
        W_t = w
        meta = None
    elif transform == 'RandOrth':
        W_t, Q_out, Q_in = apply_randorth(w)
        meta = (Q_out, Q_in)
    elif transform == 'Hadamard':
        W_t, m_out, m_in = apply_hadamard(w)
        meta = (m_out, m_in)
    
    flat = W_t.reshape(-1)
    prog, recon_t = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    recon = (atoms[prog.atom_ids.long()] * coeffs[prog.coeff_ids.long()]).reshape(W_t.shape)
    
    if transform == 'Raw':
        W_recon = recon
    elif transform == 'RandOrth':
        W_recon = inverse_randorth(recon, *meta)
    elif transform == 'Hadamard':
        W_recon = inverse_hadamard(recon, *meta)
    
    return {
        'mse': ((w - W_recon) ** 2).mean().item(),
        'relmse': ((w - W_recon).abs() / (w.abs() + 1e-8)).mean().item(),
    }


def main():
    print("=" * 60)
    print("M153 v2 — Transform-WAL Encoder (fast)")
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
    transforms = ['Raw', 'RandOrth', 'Hadamard']
    
    results = []
    
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            path = f"layers.{li}.self_attn.{m}"
            print(f"\n{path}")
            
            # Build atoms per transform
            for tname in transforms:
                if tname == 'Raw':
                    w_t = w
                elif tname == 'RandOrth':
                    w_t, _, _ = apply_randorth(w)
                elif tname == 'Hadamard':
                    w_t, _, _ = apply_hadamard(w)
                
                atoms = build_l0_atoms(w_t.reshape(-1), K=K, iters=1)
                coeffs = build_coeff_table(w_t.reshape(-1), atoms, C=C, iters=1)
                atoms = atoms[torch.argsort(atoms.abs())]
                
                t0 = time.time()
                r = encode(w, tname, atoms, coeffs)
                r['encode_time'] = time.time() - t0
                r['path'] = path
                r['transform'] = tname
                
                print(f"  {tname:12s}: MSE={r['mse']:.2e}  relMSE={r['relmse']:.2e}  time={r['encode_time']:.2f}s")
                results.append(r)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tname in transforms:
        mses = [r['mse'] for r in results if r['transform'] == tname]
        times = [r['encode_time'] for r in results if r['transform'] == tname]
        if mses:
            print(f"  {tname:12s}: avg MSE={sum(mses)/len(mses):.2e}  avg time={sum(times)/len(times):.2f}s")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m153_transform_wal_encoder.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
