#!/usr/bin/env python3
"""M153 — Real Transform-WAL Encoder.

Full pipeline: W → Transform(W) → WAL encode → WAL decode → inverse Transform → W_recon.
Compares Raw-WAL vs RandOrth-WAL vs FFT2-WAL vs DCT2-WAL vs Hadamard-WAL.
Uses CPU for encoding to avoid GPU OOM.
"""
import torch
import torch.nn as nn
import time
import json
import math
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1, wal_decode_v1


def rand_orthogonal_matrix(n, device='cpu'):
    """Generate random orthogonal matrix via QR decomposition."""
    H = torch.randn(n, n, device=device)
    Q, R = torch.linalg.qr(H)
    d = torch.diagonal(R, 0)
    ph = d.sign()
    Q = Q * ph.unsqueeze(0)
    return Q


def apply_randorth(W, Q_out=None, Q_in=None):
    """Apply random orthogonal transform: W' = Q_out @ W @ Q_in.T"""
    out_d, in_d = W.shape
    if Q_out is None:
        Q_out = rand_orthogonal_matrix(out_d, device=W.device)
    if Q_in is None:
        Q_in = rand_orthogonal_matrix(in_d, device=W.device)
    W_t = Q_out @ W @ Q_in.T
    return W_t, Q_out, Q_in


def inverse_randorth(W_t, Q_out, Q_in):
    return Q_out.T @ W_t @ Q_in


def apply_fft2(W):
    W_complex = torch.fft.fft2(W.to(torch.complex64))
    return torch.view_as_real(W_complex), None, None


def inverse_fft2(W_t, *_):
    W_complex = torch.view_as_complex(W_t.to(torch.float32))
    return torch.fft.ifft2(W_complex).real


def apply_dct2(W):
    from scipy.fft import dctn
    W_np = W.cpu().numpy()
    W_dct = dctn(W_np, type=2, norm='ortho')
    return torch.from_numpy(W_dct).to(W.device), None, None


def inverse_dct2(W_t, *_):
    from scipy.fft import idctn
    W_np = W_t.cpu().numpy()
    W_inv = idctn(W_np, type=2, norm='ortho')
    return torch.from_numpy(W_inv).to(W_t.device)


def hadamard_matrix(n, device='cpu'):
    assert n & (n - 1) == 0, f"n={n} not power of 2"
    if n == 1:
        return torch.ones(1, 1, device=device)
    H = hadamard_matrix(n // 2, device=device)
    return torch.cat([torch.cat([H, H], dim=1),
                      torch.cat([H, -H], dim=1)], dim=0)


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


TRANSFORMS = {
    'Raw': (None, None),
    'RandOrth': (apply_randorth, inverse_randorth),
    'FFT2': (apply_fft2, inverse_fft2),
    'DCT2': (apply_dct2, inverse_dct2),
    'Hadamard': (apply_hadamard_ortho, inverse_hadamard_ortho),
}


def encode_transform_wal(W, transform_name, K=256, C=16, iters=2):
    apply_fn, inverse_fn = TRANSFORMS[transform_name]
    
    t0 = time.time()
    
    if apply_fn is None:
        W_t = W
        meta = None
    else:
        W_t, *meta = apply_fn(W)
    
    flat = W_t.reshape(-1)
    
    atoms = build_l0_atoms(flat, K=K, iters=iters)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=iters)
    atoms = atoms[torch.argsort(atoms.abs())]
    
    prog, recon_t = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    
    W_recon_t = wal_decode_v1(prog, None, coeffs)
    W_recon_t = W_recon_t.reshape(W_t.shape)
    
    if inverse_fn is None:
        W_recon = W_recon_t
    else:
        W_recon = inverse_fn(W_recon_t, *meta)
    
    encode_time = time.time() - t0
    
    mse = ((W - W_recon) ** 2).mean().item()
    relmse = ((W - W_recon).abs() / (W.abs() + 1e-8)).mean().item()
    max_err = (W - W_recon).abs().max().item()
    
    counts = torch.bincount(prog.atom_ids.long(), minlength=K).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    atom_entropy = -(probs * torch.log2(probs)).sum().item() / math.log2(K)
    
    return {
        'mse': mse,
        'relmse': relmse,
        'max_err': max_err,
        'encode_time': encode_time,
        'atom_entropy': atom_entropy,
    }


def main():
    print("=" * 60)
    print("M153 — Real Transform-WAL Encoder")
    print("=" * 60)
    
    # Load model
    print("\nLoading Llama-3.1-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    layers_to_test = [0, 8, 16, 24, 31]
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    results = []
    
    for layer_idx in layers_to_test:
        layer = model.model.layers[layer_idx]
        for mod_name in modules:
            if hasattr(layer.self_attn, mod_name):
                w = getattr(layer.self_attn, mod_name).weight.data.float().cpu()
                path = f"layers.{layer_idx}.self_attn.{mod_name}"
            elif hasattr(layer.mlp, mod_name):
                w = getattr(layer.mlp, mod_name).weight.data.float().cpu()
                path = f"layers.{layer_idx}.mlp.{mod_name}"
            else:
                continue
            
            print(f"\n{path} [{w.shape[0]}×{w.shape[1]}]")
            
            for tname in ['Raw', 'RandOrth', 'FFT2', 'DCT2', 'Hadamard']:
                try:
                    result = encode_transform_wal(w, tname, K=256, C=16, iters=2)
                    print(f"  {tname:12s}: MSE={result['mse']:.2e}  relMSE={result['relmse']:.2e}  "
                          f"max_err={result['max_err']:.2e}  time={result['encode_time']:.2f}s  "
                          f"entropy={result['atom_entropy']:.4f}")
                    
                    results.append({
                        'path': path,
                        'transform': tname,
                        **result,
                    })
                except Exception as e:
                    print(f"  {tname:12s}: FAILED — {e}")
                    results.append({
                        'path': path,
                        'transform': tname,
                        'error': str(e),
                    })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Average MSE by Transform")
    print("=" * 60)
    
    for tname in ['Raw', 'RandOrth', 'FFT2', 'DCT2', 'Hadamard']:
        mses = [r['mse'] for r in results if r.get('transform') == tname and 'mse' in r]
        relmses = [r['relmse'] for r in results if r.get('transform') == tname and 'relmse' in r]
        if mses:
            avg_mse = sum(mses) / len(mses)
            avg_relmse = sum(relmses) / len(relmses)
            print(f"  {tname:12s}: avg MSE = {avg_mse:.2e}  avg relMSE = {avg_relmse:.2e}  (n={len(mses)})")
    
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m153_transform_wal_encoder.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
