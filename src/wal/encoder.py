"""WAL-0 Encoder: greedy residual encoding with k-means atoms."""
import torch
from typing import Tuple
from .isa import ProgramBuffer


def build_atoms_kmeans(weights: torch.Tensor, K: int, iters: int = 5, device=None) -> torch.Tensor:
    """Learn K atoms via mini-batch k-means++ on GPU.
    
    Args:
        weights: 1D tensor of weight values (sample)
        K: number of atoms
        iters: k-means iterations
        device: GPU device (defaults to weights.device)
    
    Returns:
        atoms: [K] float32 tensor on CPU
    """
    if device is None:
        device = weights.device
    samples = weights.to(device).float()
    
    # K-means++ initialization
    atoms = torch.zeros(K, device=device, dtype=torch.float32)
    atoms[0] = samples[torch.randint(0, samples.numel(), (1,), device=device)]
    for k in range(1, K):
        dists = (samples.unsqueeze(1) - atoms[:k].unsqueeze(0)).abs().min(dim=1)[0]
        probs = dists / dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx = idx.clamp_max(samples.numel() - 1)
        atoms[k] = samples[idx]
    
    # K-means iterations
    batch_size = 262144
    for _ in range(iters):
        assignments = torch.empty(samples.numel(), dtype=torch.int64, device=device)
        for start in range(0, samples.numel(), batch_size):
            end = min(start + batch_size, samples.numel())
            assignments[start:end] = (
                samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)
            ).abs().argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean()
    
    return atoms.cpu()


def wal_encode_scalar(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    lmax: int,
    batch: int = 524288,
) -> Tuple[ProgramBuffer, torch.Tensor]:
    """Greedy ternary residual encoding.
    
    Args:
        weights: [N] normalized weights
        atoms: [K] atom table
        lmax: max program length
        batch: batch size for GPU encoding
    
    Returns:
        prog: ProgramBuffer with encoded programs
        recon: [N] reconstructed weights
    """
    N = weights.numel()
    device = weights.device
    atoms_gpu = atoms.to(device)
    
    indices = torch.zeros(N, lmax, dtype=torch.uint8, device='cpu')
    signs = torch.zeros(N, lmax, dtype=torch.int8, device='cpu')
    residual = weights.clone()
    
    for step in range(lmax):
        best_ids = torch.empty(N, dtype=torch.int64, device=device)
        best_signs = torch.empty(N, dtype=torch.int64, device=device)
        
        for start in range(0, N, batch):
            end = min(start + batch, N)
            b = residual[start:end]
            
            # Score positive, negative, zero
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2  # [batch, K]
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2  # [batch, K]
            sz = b.unsqueeze(1) ** 2  # [batch, 1]
            
            # Best positive and negative
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            
            # Compare: pos vs neg vs zero
            use_pos = (mp < mn) & (mp < sz.squeeze(1))
            use_neg = (mn <= mp) & (mn < sz.squeeze(1))
            
            best_ids[start:end] = torch.where(use_pos, ip, torch.where(use_neg, in_, torch.zeros_like(ip)))
            best_signs[start:end] = torch.where(use_pos, 1, torch.where(use_neg, -1, 0))
        
        indices[:, step] = best_ids.cpu().to(torch.uint8)
        signs[:, step] = best_signs.cpu().to(torch.int8)
        
        # Update residual
        step_recon = atoms_gpu[best_ids] * best_signs.float()
        residual -= step_recon
    
    prog = ProgramBuffer(indices, signs, lmax)
    
    # Final reconstruction (inline to avoid circular import)
    atoms_gpu = atoms.to(device)
    gathered = atoms_gpu[indices.to(device).long()] * signs.to(device).float()
    recon = gathered.sum(dim=1)
    
    return prog, recon


def wal_encode_scalar_gpu(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    lmax: int,
    batch: int = 524288,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy ternary residual encoding — fully GPU-native, no CPU copy.
    
    Args:
        weights: [N] normalized weights
        atoms: [K] atom table
        lmax: max program length
        batch: batch size for GPU encoding
    
    Returns:
        indices: [N, lmax] uint8 on GPU
        signs: [N, lmax] int8 on GPU
        recon: [N] reconstructed weights on GPU
    """
    N = weights.numel()
    device = weights.device
    atoms_gpu = atoms.to(device)
    
    indices = torch.zeros(N, lmax, dtype=torch.uint8, device=device)
    signs = torch.zeros(N, lmax, dtype=torch.int8, device=device)
    residual = weights.clone()
    
    for step in range(lmax):
        best_ids = torch.empty(N, dtype=torch.int64, device=device)
        best_signs = torch.empty(N, dtype=torch.int64, device=device)
        
        for start in range(0, N, batch):
            end = min(start + batch, N)
            b = residual[start:end]
            
            sp = (b.unsqueeze(1) - atoms_gpu.unsqueeze(0)) ** 2
            sn = (b.unsqueeze(1) + atoms_gpu.unsqueeze(0)) ** 2
            sz = b.unsqueeze(1) ** 2
            
            mp, ip = sp.min(dim=1)
            mn, in_ = sn.min(dim=1)
            
            use_pos = (mp < mn) & (mp < sz.squeeze(1))
            use_neg = (mn <= mp) & (mn < sz.squeeze(1))
            
            best_ids[start:end] = torch.where(use_pos, ip, torch.where(use_neg, in_, torch.zeros_like(ip)))
            best_signs[start:end] = torch.where(use_pos, 1, torch.where(use_neg, -1, 0))
        
        indices[:, step] = best_ids.to(torch.uint8)
        signs[:, step] = best_signs.to(torch.int8)
        
        step_recon = atoms_gpu[best_ids] * best_signs.float()
        residual -= step_recon
    
    # Final reconstruction
    gathered = atoms_gpu[indices.long()] * signs.float()
    recon = gathered.sum(dim=1)
    
    return indices, signs, recon


def wal_decode_scalar_torch(prog: ProgramBuffer, atoms: torch.Tensor) -> torch.Tensor:
    """Decode WAL-0 programs using PyTorch (CPU/GPU).
    
    Args:
        prog: ProgramBuffer [N, lmax]
        atoms: [K] atom table
    
    Returns:
        recon: [N] reconstructed weights
    """
    device = atoms.device
    indices = prog.indices.to(device).long()
    signs = prog.signs.to(device).float()
    
    # Gather atoms and apply signs
    gathered = atoms[indices] * signs  # [N, lmax]
    recon = gathered.sum(dim=1)  # [N]
    return recon
