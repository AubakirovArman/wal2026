#!/usr/bin/env python3
"""M49: WAL-1 Vector Atoms prototype — row-wise encoding vs WAL-0 scalar."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal import wal_encode_scalar, build_atoms_kmeans, wal_decode_scalar_torch
from wal.isa import ProgramBuffer


def build_vector_atoms_kmeans(rows: torch.Tensor, K: int, iters: int = 5, device=None) -> torch.Tensor:
    """Learn K vector atoms via k-means on rows.
    
    Args:
        rows: [M, D] matrix rows as samples
        K: number of vector atoms
        iters: k-means iterations
        device: GPU device
    
    Returns:
        atoms: [K, D] float32
    """
    if device is None:
        device = rows.device
    samples = rows.to(device).float()  # [M, D]
    M, D = samples.shape
    
    # K-means++ init (batched distances)
    atoms = torch.zeros(K, D, device=device, dtype=torch.float32)
    atoms[0] = samples[torch.randint(0, M, (1,), device=device)]
    dist_batch = 1024
    for k in range(1, K):
        dists = torch.empty(M, device=device)
        for start in range(0, M, dist_batch):
            end = min(start + dist_batch, M)
            dists[start:end] = (
                samples[start:end].unsqueeze(1) - atoms[:k].unsqueeze(0)
            ).pow(2).sum(dim=-1).min(dim=1)[0]
        probs = dists / dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx = idx.clamp_max(M - 1)
        atoms[k] = samples[idx]
    
    # K-means iterations
    assign_batch = 1024
    for _ in range(iters):
        assignments = torch.empty(M, dtype=torch.int64, device=device)
        for start in range(0, M, assign_batch):
            end = min(start + assign_batch, M)
            assignments[start:end] = (
                samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)
            ).pow(2).sum(dim=-1).argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean(dim=0)
    
    return atoms


def wal_encode_vector(
    rows: torch.Tensor,
    atoms: torch.Tensor,
    lmax: int,
    batch: int = 4096,
):
    """Row-wise vector encoding.
    
    Args:
        rows: [M, D] normalized rows
        atoms: [K, D] vector atoms
        lmax: max program length
        batch: batch size
    
    Returns:
        prog: ProgramBuffer [M, lmax]
        recon: [M, D] reconstructed rows
    """
    M, D = rows.shape
    device = rows.device
    atoms_gpu = atoms.to(device)  # [K, D]
    
    indices = torch.zeros(M, lmax, dtype=torch.uint8, device='cpu')
    signs = torch.zeros(M, lmax, dtype=torch.int8, device='cpu')
    residual = rows.clone()  # [M, D]
    
    for step in range(lmax):
        best_ids = torch.empty(M, dtype=torch.int64, device=device)
        best_signs = torch.empty(M, dtype=torch.int64, device=device)
        
        for start in range(0, M, batch):
            end = min(start + batch, M)
            b = residual[start:end]  # [batch, D]
            
            # Score: ||b - atom||^2, ||b + atom||^2, ||b||^2
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)).pow(2).sum(dim=-1)  # [batch, K]
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)).pow(2).sum(dim=-1)  # [batch, K]
            sz = b.pow(2).sum(dim=-1, keepdim=True)  # [batch, 1]
            
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            
            use_pos = (mp < mn) & (mp < sz.squeeze(1))
            use_neg = (mn <= mp) & (mn < sz.squeeze(1))
            
            best_ids[start:end] = torch.where(use_pos, ip, torch.where(use_neg, in_, torch.zeros_like(ip)))
            best_signs[start:end] = torch.where(use_pos, 1, torch.where(use_neg, -1, 0))
        
        indices[:, step] = best_ids.cpu().to(torch.uint8)
        signs[:, step] = best_signs.cpu().to(torch.int8)
        
        step_recon = atoms_gpu[best_ids] * best_signs.unsqueeze(1).float()  # [M, D]
        residual -= step_recon
    
    prog = ProgramBuffer(indices, signs, lmax)
    
    # Decode
    atoms_gpu = atoms.to(device)
    gathered = atoms_gpu[indices.to(device).long()] * signs.to(device).float().unsqueeze(-1)  # [M, lmax, D]
    recon = gathered.sum(dim=1)  # [M, D]
    
    return prog, recon


def benchmark():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    M, D = 1024, 4096  # Reduced for GPU memory (full Gemma down_proj would OOM)
    K = 128
    lmax = 2
    
    print("=" * 60)
    print("M49: WAL-1 Vector Atoms Prototype")
    print("=" * 60)
    print(f"Matrix: [{M}, {D}], K={K}, lmax={lmax}, device={device}")
    
    # Generate synthetic weights with row-wise structure
    torch.manual_seed(42)
    true_atoms = torch.randn(16, D, device=device)
    weights = torch.zeros(M, D, device=device)
    for i in range(M):
        idx = torch.randint(0, 16, (2,), device=device)
        weights[i] = true_atoms[idx[0]] * 0.7 + true_atoms[idx[1]] * 0.3 + torch.randn(D, device=device) * 0.1
    
    # Normalize per row (standard)
    row_scale = weights.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = weights / row_scale
    
    # === WAL-0 Scalar ===
    print(f"\n[1] WAL-0 Scalar encode...")
    flat = w_norm.reshape(-1)
    SAMPLE_SIZE = 1_000_000
    if flat.numel() > SAMPLE_SIZE:
        idx_samp = torch.randperm(flat.numel())[:SAMPLE_SIZE]
        samples_s = flat[idx_samp]
    else:
        samples_s = flat
    t0 = time.time()
    atoms_s = build_atoms_kmeans(samples_s, K, iters=5, device=device)
    atoms_s = atoms_s.to(device)
    prog_s, recon_s_flat = wal_encode_scalar(flat, atoms_s, lmax)
    t_s = time.time() - t0
    recon_s = recon_s_flat.reshape(M, D) * row_scale
    
    rel_mse_s = ((weights - recon_s) ** 2).mean() / (weights ** 2).mean()
    print(f"    Time: {t_s:.2f}s")
    print(f"    relMSE: {rel_mse_s.item():.6f}")
    
    # === WAL-1 Vector ===
    print(f"\n[2] WAL-1 Vector encode...")
    t0 = time.time()
    atoms_v = build_vector_atoms_kmeans(w_norm, K, iters=5, device=device)
    atoms_v = atoms_v.to(device)
    prog_v, recon_v_norm = wal_encode_vector(w_norm, atoms_v, lmax)
    t_v = time.time() - t0
    recon_v = recon_v_norm * row_scale
    
    rel_mse_v = ((weights - recon_v) ** 2).mean() / (weights ** 2).mean()
    print(f"    Time: {t_v:.2f}s")
    print(f"    relMSE: {rel_mse_v.item():.6f}")
    
    # === Output quality (matmul) ===
    print(f"\n[3] Output quality test (matmul with random input)...")
    x = torch.randn(1, 128, M, device=device)  # [batch, seq, M]
    
    out_orig = torch.matmul(x, weights)  # [1, 128, D]
    out_s = torch.matmul(x, recon_s)
    out_v = torch.matmul(x, recon_v)
    
    rel_mse_out_s = ((out_orig - out_s) ** 2).mean() / (out_orig ** 2).mean()
    rel_mse_out_v = ((out_orig - out_v) ** 2).mean() / (out_orig ** 2).mean()
    
    print(f"    WAL-0 output relMSE: {rel_mse_out_s.item():.6f}")
    print(f"    WAL-1 output relMSE: {rel_mse_out_v.item():.6f}")
    print(f"    Improvement: {rel_mse_out_s.item() / rel_mse_out_v.item():.1f}x")
    
    # === Storage comparison ===
    print(f"\n[4] Storage comparison...")
    orig_bytes = weights.numel() * 2  # bf16
    
    # WAL-0: K floats + N*lmax bytes (idx+sign)
    wal0_bytes = K * 4 + prog_s.N * prog_s.lmax * 2
    wal0_ratio = orig_bytes / wal0_bytes
    
    # WAL-1: K*D floats + M*lmax bytes
    wal1_bytes = K * D * 4 + prog_v.N * prog_v.lmax * 2
    wal1_ratio = orig_bytes / wal1_bytes
    
    print(f"    Original:     {orig_bytes / 1e6:.1f} MB")
    print(f"    WAL-0 scalar: {wal0_bytes / 1e6:.1f} MB  ({wal0_ratio:.2f}x)")
    print(f"    WAL-1 vector: {wal1_bytes / 1e6:.1f} MB  ({wal1_ratio:.2f}x)")
    
    print("\n" + "=" * 60)
    print("M49: DONE")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
