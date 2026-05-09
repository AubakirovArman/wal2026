#!/usr/bin/env python3
"""M50: WAL-1 with SVD-based atoms on real Llama 70B weights."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal import wal_encode_scalar, build_atoms_kmeans


def svd_atoms(weight: torch.Tensor, K: int):
    """Extract top-K right singular vectors as atoms."""
    # weight: [M, D], SVD: W = U S Vh
    # Vh: [min(M,D), D], rows are right singular vectors
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    # Top-K singular vectors
    atoms = Vh[:K, :]  # [K, D]
    return atoms, S[:K]


def wal_encode_vector_svd(
    rows: torch.Tensor,
    atoms: torch.Tensor,
    lmax: int,
    batch: int = 1024,
):
    """Row-wise encoding with SVD atoms.
    
    Args:
        rows: [M, D] normalized rows
        atoms: [K, D] orthonormal atoms
        lmax: max program length
    
    Returns:
        recon: [M, D] reconstructed rows
    """
    M, D = rows.shape
    device = rows.device
    atoms_gpu = atoms.to(device)  # [K, D]
    
    # For orthonormal atoms, optimal coefficients are dot products
    # But we want ternary coefficients, so we do greedy residual
    indices = torch.zeros(M, lmax, dtype=torch.uint8, device='cpu')
    signs = torch.zeros(M, lmax, dtype=torch.int8, device='cpu')
    residual = rows.clone()  # [M, D]
    
    for step in range(lmax):
        best_ids = torch.empty(M, dtype=torch.int64, device=device)
        best_signs = torch.empty(M, dtype=torch.int64, device=device)
        
        for start in range(0, M, batch):
            end = min(start + batch, M)
            b = residual[start:end]  # [batch, D]
            
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)).pow(2).sum(dim=-1)
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)).pow(2).sum(dim=-1)
            sz = b.pow(2).sum(dim=-1, keepdim=True)
            
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            
            use_pos = (mp < mn) & (mp < sz.squeeze(1))
            use_neg = (mn <= mp) & (mn < sz.squeeze(1))
            
            best_ids[start:end] = torch.where(use_pos, ip, torch.where(use_neg, in_, torch.zeros_like(ip)))
            best_signs[start:end] = torch.where(use_pos, 1, torch.where(use_neg, -1, 0))
        
        indices[:, step] = best_ids.cpu().to(torch.uint8)
        signs[:, step] = best_signs.cpu().to(torch.int8)
        
        step_recon = atoms_gpu[best_ids] * best_signs.unsqueeze(1).float()
        residual -= step_recon
    
    # Decode
    gathered = atoms_gpu[indices.to(device).long()] * signs.to(device).float().unsqueeze(-1)
    recon = gathered.sum(dim=1)
    return recon


def test():
    device = torch.device('cuda:2')
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    layer_idx = 50
    param_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    
    print("=" * 60)
    print("M50: WAL-1 with SVD Atoms on Real 70B Weights")
    print("=" * 60)
    
    print(f"\n[1] Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    
    param = dict(model.named_parameters())[param_name]
    print(f"\n[2] Parameter: {param_name}")
    print(f"    Shape: {tuple(param.shape)}")
    
    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    M, D = w_norm.shape
    
    # Ensure all on same device for encoding
    encode_device = device
    w = w.to(encode_device)
    row_scale = row_scale.to(encode_device)
    w_norm = w_norm.to(encode_device)
    
    K = 128
    lmax = 2
    
    # === WAL-0 Scalar ===
    print(f"\n[3] WAL-0 Scalar (baseline)...")
    flat = w_norm.reshape(-1)
    SAMPLE_SIZE = 1_000_000
    if flat.numel() > SAMPLE_SIZE:
        idx_samp = torch.randperm(flat.numel())[:SAMPLE_SIZE]
        samples = flat[idx_samp]
    else:
        samples = flat
    
    t0 = time.time()
    atoms_s = build_atoms_kmeans(samples, K, iters=5, device=device)
    atoms_s = atoms_s.to(device)
    from wal import wal_encode_scalar
    prog_s, recon_s_flat = wal_encode_scalar(flat.to(device), atoms_s, lmax)
    recon_s = recon_s_flat.reshape(M, D) * row_scale
    t_s = time.time() - t0
    
    rel_mse_s = ((w - recon_s) ** 2).mean() / (w ** 2).mean()
    print(f"    Time: {t_s:.2f}s, relMSE: {rel_mse_s.item():.8f}")
    
    # === WAL-1 SVD ===
    print(f"\n[4] WAL-1 SVD atoms...")
    t0 = time.time()
    atoms_v, sing_vals = svd_atoms(w_norm, K)
    atoms_v = atoms_v.to(device)
    recon_v_norm = wal_encode_vector_svd(w_norm, atoms_v, lmax)
    recon_v = recon_v_norm * row_scale
    t_v = time.time() - t0
    
    rel_mse_v = ((w - recon_v) ** 2).mean() / (w ** 2).mean()
    print(f"    Time: {t_v:.2f}s, relMSE: {rel_mse_v.item():.8f}")
    print(f"    Singular values (top 10): {sing_vals[:10].cpu().numpy()}")
    
    # === Matmul test ===
    print(f"\n[5] Output quality test...")
    import torch.nn.functional as F
    x = torch.randn(1, 128, D, dtype=torch.bfloat16, device=device)
    
    out_orig = F.linear(x, w.to(torch.bfloat16))
    out_s = F.linear(x, recon_s.to(torch.bfloat16))
    out_v = F.linear(x, recon_v.to(torch.bfloat16))
    
    out_rel_s = ((out_orig - out_s) ** 2).mean() / (out_orig ** 2).mean()
    out_rel_v = ((out_orig - out_v) ** 2).mean() / (out_orig ** 2).mean()
    out_corr_s = torch.corrcoef(torch.stack([out_orig.reshape(-1), out_s.reshape(-1)]))[0, 1]
    out_corr_v = torch.corrcoef(torch.stack([out_orig.reshape(-1), out_v.reshape(-1)]))[0, 1]
    
    print(f"    WAL-0 output relMSE: {out_rel_s.item():.8f}, corr: {out_corr_s.item():.6f}")
    print(f"    WAL-1 output relMSE: {out_rel_v.item():.8f}, corr: {out_corr_v.item():.6f}")
    
    # Storage
    print(f"\n[6] Storage...")
    orig_bytes = w.numel() * 2
    wal0_bytes = K * 4 + prog_s.N * prog_s.lmax * 2
    wal1_bytes = K * D * 4 + M * lmax * 2
    print(f"    Original:     {orig_bytes / 1e6:.1f} MB")
    print(f"    WAL-0 scalar: {wal0_bytes / 1e6:.1f} MB ({orig_bytes / wal0_bytes:.2f}x)")
    print(f"    WAL-1 SVD:    {wal1_bytes / 1e6:.1f} MB ({orig_bytes / wal1_bytes:.2f}x)")
    
    print("\n" + "=" * 60)
    print("M50: DONE")
    print("=" * 60)


if __name__ == "__main__":
    test()
