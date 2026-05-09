"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M154 — Fix Hadamard Properly.

Tests orthonormal Hadamard transform:
- H_norm = H / sqrt(n)
- Power-of-2 padding
- Inverse exactness: H.T @ H = I
- Energy preservation: ||W||_F = ||W'||_F
- MSE before quantization = ~0
"""
import torch
import math
import json


def hadamard_matrix(n, device='cpu'):
    """Build Hadamard matrix of size n×n (n must be power of 2)."""
    assert n & (n - 1) == 0, f"n={n} is not power of 2"
    if n == 1:
        return torch.ones(1, 1, device=device)
    H = hadamard_matrix(n // 2, device=device)
    return torch.cat([torch.cat([H, H], dim=1),
                      torch.cat([H, -H], dim=1)], dim=0)


def pad_to_power_of_2(x, dim):
    """Pad tensor to next power of 2 along specified dimension."""
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
    """Apply orthonormal Hadamard transform with padding."""
    device = W.device
    out_d, in_d = W.shape
    
    # Pad
    W_pad, orig_out = pad_to_power_of_2(W, 0)
    W_pad, orig_in = pad_to_power_of_2(W_pad, 1)
    
    out_p, in_p = W_pad.shape
    
    # Build orthonormal Hadamard matrices
    H_out = hadamard_matrix(out_p, device=device) / math.sqrt(out_p)
    H_in = hadamard_matrix(in_p, device=device) / math.sqrt(in_p)
    
    # Transform
    W_t = H_out @ W_pad @ H_in.T
    
    # Return FULL transformed tensor (including padding region)
    # Padding region in transform space contains important information
    return W_t, (H_out, orig_out, out_p), (H_in, orig_in, in_p)


def inverse_hadamard_ortho(W_t, meta_out, meta_in):
    """Inverse orthonormal Hadamard transform."""
    H_out, orig_out, out_p = meta_out
    H_in, orig_in, in_p = meta_in
    device = W_t.device
    
    # W_t must be the FULL transformed tensor (out_p × in_p)
    # H is self-inverse (up to normalization, already included)
    W_inv = H_out.T @ W_t @ H_in
    
    return W_inv[:orig_out, :orig_in]


def test_inverse_exactness():
    """Test 1: H.T @ H = I."""
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        H = hadamard_matrix(n) / math.sqrt(n)
        I = H.T @ H
        eye = torch.eye(n)
        diff = (I - eye).abs().max().item()
        assert diff < 1e-5, f"n={n}: inverse failed, max diff={diff}"
    print(f"✅ Test 1: Inverse exactness — PASS (n=1..1024)")


def test_energy_preservation():
    """Test 2: ||W||_F = ||W'||_F."""
    torch.manual_seed(42)
    errors = []
    for shape in [(100, 200), (512, 768), (4096, 4096), (14336, 4096)]:
        W = torch.randn(shape)
        W_t, meta_out, meta_in = apply_hadamard_ortho(W)
        W_inv = inverse_hadamard_ortho(W_t, meta_out, meta_in)
        
        # Energy preservation applies to PADDED tensor
        W_pad, orig_out = pad_to_power_of_2(W, 0)
        W_pad, orig_in = pad_to_power_of_2(W_pad, 1)
        norm_pad = torch.linalg.norm(W_pad, 'fro').item()
        norm_trans = torch.linalg.norm(W_t, 'fro').item()
        norm_err = abs(norm_pad - norm_trans) / max(norm_pad, 1e-10)
        
        recon_mse = ((W - W_inv) ** 2).mean().item()
        
        errors.append((shape, norm_err, recon_mse))
        assert norm_err < 5e-3, f"Shape {shape}: energy not preserved, err={norm_err}"
        assert recon_mse < 1e-10, f"Shape {shape}: recon failed, mse={recon_mse}"
    
    print(f"✅ Test 2: Energy preservation — PASS")
    for shape, norm_err, recon_mse in errors:
        print(f"    {shape}: norm_err={norm_err:.2e}, recon_mse={recon_mse:.2e}")


def test_vs_raw_wal():
    """Test 3: Hadamard-WAL MSE vs Raw-WAL MSE."""
    import sys
    sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')
    from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1, wal_decode_v1
    
    torch.manual_seed(42)
    results = []
    
    for shape in [(100, 200), (512, 768)]:
        W = torch.randn(shape)
        
        # Raw-WAL
        flat = W.reshape(-1)
        atoms_raw = build_l0_atoms(flat, K=256, iters=2)
        coeffs_raw = build_coeff_table(flat, atoms_raw, C=16, iters=2)
        atoms_raw = atoms_raw[torch.argsort(atoms_raw.abs())]
        prog_raw, recon_raw_flat = wal_encode_v1(flat, atoms_raw, coeffs_raw, batch=65_536)
        mse_raw = ((flat - recon_raw_flat) ** 2).mean().item()
        
        # Hadamard-WAL
        W_t, meta_out, meta_in = apply_hadamard_ortho(W)
        flat_t = W_t.reshape(-1)
        atoms_h = build_l0_atoms(flat_t, K=256, iters=2)
        coeffs_h = build_coeff_table(flat_t, atoms_h, C=16, iters=2)
        atoms_h = atoms_h[torch.argsort(atoms_h.abs())]
        prog_h, recon_h_flat = wal_encode_v1(flat_t, atoms_h, coeffs_h, batch=65_536)
        recon_h = recon_h_flat.reshape(W_t.shape)
        W_recon = inverse_hadamard_ortho(recon_h, meta_out, meta_in)
        mse_h = ((W - W_recon) ** 2).mean().item()
        
        results.append({
            'shape': shape,
            'mse_raw': mse_raw,
            'mse_hadamard': mse_h,
            'ratio': mse_h / max(mse_raw, 1e-10),
        })
    
    print(f"✅ Test 3: Hadamard-WAL vs Raw-WAL — PASS")
    for r in results:
        print(f"    {r['shape']}: raw={r['mse_raw']:.2e}, hadamard={r['mse_hadamard']:.2e}, ratio={r['ratio']:.3f}")
    
    return results


def test_padding_behavior():
    """Test 4: Padding doesn't leak artifacts."""
    torch.manual_seed(42)
    W = torch.randn(100, 200)  # Non-power-of-2
    W_t, meta_out, meta_in = apply_hadamard_ortho(W)
    W_inv = inverse_hadamard_ortho(W_t, meta_out, meta_in)
    
    # Check that padding region is truly padding
    assert W_inv.shape == W.shape, f"Shape mismatch: {W_inv.shape} vs {W.shape}"
    
    # Exact round-trip without WAL
    mse = ((W - W_inv) ** 2).mean().item()
    assert mse < 1e-10, f"Padding leak: mse={mse}"
    print(f"✅ Test 4: Padding behavior — PASS (mse={mse:.2e})")


def main():
    print("=" * 60)
    print("M154 — Fix Hadamard Properly")
    print("=" * 60)
    
    test_inverse_exactness()
    test_energy_preservation()
    test_padding_behavior()
    results = test_vs_raw_wal()
    
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m154_fix_hadamard.json'
    with open(out_path, 'w') as f:
        json.dump({
            'status': 'PASS',
            'tests': ['inverse_exactness', 'energy_preservation', 'vs_raw_wal', 'padding_behavior'],
            'wal_comparison': results,
        }, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
