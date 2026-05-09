"""WAL v2 Encoder: Single-call programs with continuous coefficients.

Encoding:
    weight = atom[atom_id] * coeff[coeff_id] + residual

Algorithm:
    1. Build atom table via k-means (K atoms)
    2. Build coefficient table via Lloyd-Max on ratios w / best_atom (C levels)
    3. For each weight: joint search for (atom_id, coeff_id) minimizing |w - atom*coeff|
    4. Optional: add residual if error > threshold
"""
import torch
from typing import Tuple
from .isa import ProgramBufferV2, AtomTable, CoeffTable


def build_atoms_kmeans_v2(weights: torch.Tensor, K: int, iters: int = 5, device=None) -> torch.Tensor:
    """Build atom table via k-means++.
    
    Args:
        weights: 1D tensor of normalized weight values
        K: number of atoms
        iters: k-means iterations
        device: GPU device
        
    Returns:
        atoms: [K] float32 on CPU
    """
    if device is None:
        device = weights.device
    samples = weights.to(device).float().flatten()
    
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


def build_coeff_table(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    C: int,
    iters: int = 5,
    device=None,
    batch: int = 2_097_152,
    max_samples: int = 2_000_000,
) -> torch.Tensor:
    """Build coefficient table via Lloyd-Max on ratios w / atom.
    
    For each weight, find nearest atom, then ratio = w / atom.
    Run Lloyd-Max on sampled ratios to get C quantization levels.
    
    Args:
        weights: 1D tensor of normalized weight values
        atoms: [K] atom table
        C: number of coefficient levels
        iters: Lloyd-Max iterations
        device: GPU device
        batch: batch size for atom assignment
        max_samples: max ratios to use for Lloyd-Max (speed/quality tradeoff)
        
    Returns:
        coeffs: [C] float32 (sorted ascending)
    """
    if device is None:
        device = weights.device
    
    w = weights.to(device).float().flatten()
    a = atoms.to(device)
    N = w.numel()
    
    # Find nearest atom for each weight (batched, all on GPU)
    best_atom_vals = torch.empty(N, device=device, dtype=torch.float32)
    for start in range(0, N, batch):
        end = min(start + batch, N)
        b = w[start:end]
        best_k = (b.unsqueeze(1) - a.unsqueeze(0)).abs().argmin(dim=1)
        best_atom_vals[start:end] = a[best_k]
    
    # Ratios (avoid division by near-zero)
    ratios = w / best_atom_vals.clamp_min(1e-8)
    ratios = ratios.clamp(-10.0, 10.0)
    
    # Sample ratios for Lloyd-Max (faster, stable)
    if ratios.numel() > max_samples:
        sample_idx = torch.randperm(ratios.numel(), device=device)[:max_samples]
        sample_ratios = ratios[sample_idx]
    else:
        sample_ratios = ratios
    
    # Lloyd-Max on sampled ratios (all on GPU)
    r_min, r_max = sample_ratios.min().item(), sample_ratios.max().item()
    coeffs = torch.linspace(r_min, r_max, C, device=device)
    
    for _ in range(iters):
        dists = (sample_ratios.unsqueeze(1) - coeffs.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        for c in range(C):
            mask = assignments == c
            if mask.any():
                coeffs[c] = sample_ratios[mask].mean()
    
    return coeffs.cpu().sort().values


def _encode_batch(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    coeffs: torch.Tensor,
    residual_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a batch of weights.
    
    Uses ratio-based coeff selection for speed:
    1. For each weight and atom, compute best coeff via ratio quantization
    2. Select (atom, coeff) pair with minimum error
    
    Args:
        weights: [B] float32
        atoms: [K] float32
        coeffs: [C] float32 (sorted)
        residual_threshold: add residual if |error| > threshold
        
    Returns:
        atom_ids: [B] uint8
        coeff_ids: [B] uint8
        residuals: [B] float32
        has_residual: [B] bool
    """
    B = weights.shape[0]
    K = atoms.shape[0]
    C = coeffs.shape[0]
    device = weights.device
    
    # Step 1: For each weight and atom, find best coeff via ratio
    # ratios: [B, K] = w[i] / atom[k]
    ratios = weights.unsqueeze(1) / atoms.unsqueeze(0).clamp_min(1e-8)
    ratios = ratios.clamp(-10.0, 10.0)
    
    # Find nearest coeff for each ratio using bucketize + neighbor check
    insert = torch.bucketize(ratios, coeffs)  # [B, K]
    insert = insert.clamp(0, C - 1)
    
    # Check distances to insert and insert-1
    best_c = insert.clone()
    best_dist = (ratios - coeffs[insert]).abs()
    
    if C > 1:
        insert_m1 = (insert - 1).clamp_min(0)
        dist_m1 = (ratios - coeffs[insert_m1]).abs()
        better_m1 = dist_m1 < best_dist
        best_c = torch.where(better_m1, insert_m1, best_c)
        best_dist = torch.where(better_m1, dist_m1, best_dist)
    
    # Step 2: Reconstruct with best (atom, coeff) and compute errors
    recon = atoms.unsqueeze(0) * coeffs[best_c]  # [B, K]
    errors = (weights.unsqueeze(1) - recon) ** 2  # [B, K]
    
    # Step 3: Select best atom for each weight
    best_k = errors.argmin(dim=1)  # [B]
    best_c_for_best_k = best_c[torch.arange(B, device=device), best_k]
    
    # Step 4: Compute final reconstruction and residual
    final_recon = atoms[best_k] * coeffs[best_c_for_best_k]
    residual = weights - final_recon
    abs_residual = residual.abs()
    
    # Step 5: Apply residual threshold
    if residual_threshold > 0:
        has_residual = abs_residual > residual_threshold
        # Quantize residual to float16 range for compactness
        # For now, keep as float32 but only store where needed
        residual = torch.where(has_residual, residual, torch.zeros_like(residual))
    else:
        has_residual = torch.zeros(B, dtype=torch.bool, device=device)
        residual = torch.zeros_like(residual)
    
    return best_k.to(torch.uint8), best_c_for_best_k.to(torch.uint8), residual, has_residual


def wal_encode_v2(
    weights: torch.Tensor,
    atoms: AtomTable,
    coeffs: CoeffTable,
    residual_threshold: float = 0.0,
    batch: int = 1_048_576,
    shape: Tuple[int, ...] = None,
) -> Tuple[ProgramBufferV2, torch.Tensor]:
    """Encode weights to WAL v2 programs.
    
    Args:
        weights: [M, N] or [N] normalized weights
        atoms: AtomTable with K atoms
        coeffs: CoeffTable with C levels
        residual_threshold: add residual if |error| > threshold (0 = no residuals)
        batch: batch size for GPU encoding
        shape: Override shape (useful when weights is already flattened)
        
    Returns:
        prog: ProgramBufferV2
        recon: [N] reconstructed weights (including residuals)
    """
    original_shape = shape if shape is not None else weights.shape
    w = weights.flatten()
    N = w.numel()
    device = w.device
    
    atoms_gpu = atoms.values.to(device)
    coeffs_gpu = coeffs.values.to(device)
    
    atom_ids = torch.empty(N, dtype=torch.uint8, device='cpu')
    coeff_ids = torch.empty(N, dtype=torch.uint8, device='cpu')
    residuals = torch.empty(N, dtype=torch.float32, device='cpu')
    has_residual = torch.empty(N, dtype=torch.bool, device='cpu')
    
    for start in range(0, N, batch):
        end = min(start + batch, N)
        b = w[start:end]
        
        a, c, r, hr = _encode_batch(b, atoms_gpu, coeffs_gpu, residual_threshold)
        
        atom_ids[start:end] = a.cpu()
        coeff_ids[start:end] = c.cpu()
        residuals[start:end] = r.cpu()
        has_residual[start:end] = hr.cpu()
    
    prog = ProgramBufferV2(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residuals,
        has_residual=has_residual,
        shape=original_shape,
    )
    
    # Compute reconstruction
    recon = prog.decode(atoms, coeffs)
    
    return prog, recon
