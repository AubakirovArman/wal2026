#!/usr/bin/env python3
"""Atom Transfer — use pre-trained atoms from one model to encode another.

Phase 8: Cross-model atom reuse and transfer evaluation.
"""
import torch
from typing import Tuple, Dict
from .library import AtomLibraryEntry


def encode_with_pretrained_atoms(
    weights: torch.Tensor,
    source_entry: AtomLibraryEntry,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode weights using atoms from a library entry.
    
    Uses source atoms as initialization for k-means++ instead of
    random sampling. This can produce better atoms when source and
    target are similar (e.g., Llama-70B → Llama-8B).
    
    Args:
        weights: Target weights to encode
        source_entry: Library entry with pre-trained atoms
        device: Target device
    
    Returns:
        (adapted_atoms, adapted_coeffs, recon) where recon is reconstruction
    """
    from wal.v1 import build_coeff_table, wal_encode_v1
    
    device = device or weights.device
    K = source_entry.K
    C = source_entry.C
    N = weights.numel()
    
    # Load source atoms and adapt to target distribution
    source_atoms = source_entry.atom_tensor.to(device)
    
    # Adapt: scale source atoms to match target weight distribution
    target_std = weights.std().item()
    source_std = source_atoms.std().item()
    scale = target_std / (source_std + 1e-8)
    adapted_atoms = source_atoms * scale
    
    # Fine-tune atoms with a few k-means iterations on target
    adapted_atoms = _fine_tune_atoms(weights, adapted_atoms, iters=3, device=device)
    
    # Build coeffs using adapted atoms
    adapted_coeffs = build_coeff_table(weights, adapted_atoms, C=C, iters=3, device=device)
    
    # Encode
    prog, recon = wal_encode_v1(weights, adapted_atoms, adapted_coeffs, device=device)
    
    return adapted_atoms, adapted_coeffs, recon


def _fine_tune_atoms(weights: torch.Tensor, init_atoms: torch.Tensor,
                     iters: int = 3, batch: int = 65536, device=None) -> torch.Tensor:
    """Fine-tune atoms with k-means starting from pre-trained values.
    
    Args:
        weights: Target weights
        init_atoms: Initial atom values (pre-trained)
        iters: Number of k-means iterations
        batch: Batch size for assignments
        device: Target device
    
    Returns:
        Fine-tuned atoms
    """
    device = device or weights.device
    atoms = init_atoms.to(device).clone()
    K = atoms.numel()
    N = weights.numel()
    
    samples = weights[torch.randperm(N, device=device)[:min(N, 1_000_000)]]
    
    for _ in range(iters):
        assignments = torch.empty(samples.numel(), dtype=torch.int64, device=device)
        for start in range(0, samples.numel(), batch):
            end = min(start + batch, samples.numel())
            d = (samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs()
            assignments[start:end] = d.argmin(dim=1)
        
        for k in range(K):
            mask = assignments == k
            if mask.any():
                # Blend with original: 70% target mean, 30% original
                target_mean = samples[mask].mean()
                atoms[k] = 0.7 * target_mean + 0.3 * atoms[k]
    
    return atoms


def evaluate_transfer(
    weights: torch.Tensor,
    source_entry: AtomLibraryEntry,
    baseline_atoms: torch.Tensor,
    baseline_coeffs: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate atom transfer quality vs baseline encoding.
    
    Args:
        weights: Target weights
        source_entry: Source library entry
        baseline_atoms: Baseline atoms (from scratch k-means)
        baseline_coeffs: Baseline coeffs
    
    Returns:
        Dict with quality metrics
    """
    from wal.v1 import wal_encode_v1
    
    # Baseline encode
    _, baseline_recon = wal_encode_v1(weights, baseline_atoms, baseline_coeffs)
    baseline_mse = (weights - baseline_recon).pow(2).mean().item()
    
    # Transfer encode
    _, _, transfer_recon = encode_with_pretrained_atoms(weights, source_entry)
    transfer_mse = (weights - transfer_recon).pow(2).mean().item()
    
    return {
        'baseline_mse': baseline_mse,
        'transfer_mse': transfer_mse,
        'mse_ratio': transfer_mse / (baseline_mse + 1e-12),
        'transfer_better': transfer_mse < baseline_mse,
        'max_abs_diff_baseline': (weights - baseline_recon).abs().max().item(),
        'max_abs_diff_transfer': (weights - transfer_recon).abs().max().item(),
    }


def transfer_atoms_direct(
    weights: torch.Tensor,
    source_entry: AtomLibraryEntry,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct transfer: use source atoms scaled to target distribution.
    
    No fine-tuning. Fastest option when models are very similar.
    
    Args:
        weights: Target weights
        source_entry: Source library entry
        device: Target device
    
    Returns:
        (scaled_atoms, recon)
    """
    from wal.v1 import build_coeff_table, wal_encode_v1
    
    device = device or weights.device
    source_atoms = source_entry.atom_tensor.to(device)
    
    # Simple scale matching
    target_std = weights.std().item()
    source_std = source_atoms.std().item()
    scale = target_std / (source_std + 1e-8)
    scaled_atoms = source_atoms * scale
    
    # Build coeffs for scaled atoms
    coeffs = build_coeff_table(weights, scaled_atoms, C=source_entry.C, iters=3, device=device)
    
    # Encode
    _, recon = wal_encode_v1(weights, scaled_atoms, coeffs, device=device)
    
    return scaled_atoms, recon
