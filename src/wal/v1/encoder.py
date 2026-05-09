#!/usr/bin/env python3
"""WAL v1 encoder — builds hierarchical atoms on top of WAL v2 encoding.

Strategy:
1. Build L0 atoms via k-means (same as WAL v2)
2. Analyze atom co-occurrence in programs
3. Build L1 composite atoms from frequently co-occurring L0 pairs
4. Programs still use flattened atom IDs for fast decode
"""
import torch
from typing import Tuple, List
from .isa import AtomTableV1, AtomDef, ProgramBufferV1, CoeffTable


def build_l0_atoms(weights: torch.Tensor, K: int, iters: int = 5, device=None) -> torch.Tensor:
    """Build L0 scalar atoms via k-means++ (same as WAL v2)."""
    N = weights.numel()
    device = device or weights.device
    samples = weights[torch.randperm(N, device=device)[:min(N, 1_000_000)]]
    
    # K-means++ init
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
    batch_size = 65536
    for _ in range(iters):
        assignments = torch.empty(samples.numel(), dtype=torch.int64, device=device)
        for start in range(0, samples.numel(), batch_size):
            end = min(start + batch_size, samples.numel())
            d = (samples[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs()
            assignments[start:end] = d.argmin(dim=1)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                atoms[k] = samples[mask].mean()
    
    return atoms


def build_coeff_table(weights: torch.Tensor, atoms: torch.Tensor, C: int, 
                      iters: int = 5, device=None) -> torch.Tensor:
    """Build coefficient table via k-means on |w/atom| ratios."""
    device = device or weights.device
    N = weights.numel()
    
    # Batched nearest atom + ratio
    best_atom = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, 1_048_576):
        end = min(start + 1_048_576, N)
        d = (weights[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs()
        best_atom[start:end] = d.argmin(dim=1)
    
    ratios = (weights / atoms[best_atom]).abs()
    ratios = ratios[ratios.isfinite()]
    
    # K-means on ratios
    samples = ratios[torch.randperm(ratios.numel(), device=device)[:min(ratios.numel(), 1_000_000)]]
    coeffs = torch.zeros(C, device=device, dtype=torch.float32)
    coeffs[0] = samples[torch.randint(0, samples.numel(), (1,), device=device)]
    for c in range(1, C):
        dists = (samples.unsqueeze(1) - coeffs[:c].unsqueeze(0)).abs().min(dim=1)[0]
        probs = dists / dists.sum()
        cumprobs = probs.cumsum(dim=0)
        idx = torch.searchsorted(cumprobs, torch.rand(1, device=device))
        idx = idx.clamp_max(samples.numel() - 1)
        coeffs[c] = samples[idx]
    
    for _ in range(iters):
        assignments = (samples.unsqueeze(1) - coeffs.unsqueeze(0)).abs().argmin(dim=1)
        for c in range(C):
            mask = assignments == c
            if mask.any():
                coeffs[c] = samples[mask].mean()
    
    return coeffs


def wal_encode_v1(weights: torch.Tensor, atoms: torch.Tensor, coeffs: torch.Tensor,
                  residual_threshold: float = 0.0, batch: int = 1_048_576,
                  device=None) -> Tuple[ProgramBufferV1, torch.Tensor]:
    """Encode weights with WAL v1 (same as v2 at program level)."""
    N = weights.numel()
    device = device or weights.device
    a = atoms.to(device)
    c = coeffs.to(device)
    K = a.numel()
    C = c.numel()
    
    atom_ids = torch.empty(N, dtype=torch.uint8, device=device)
    coeff_ids = torch.empty(N, dtype=torch.uint8, device=device)
    recon = torch.empty(N, dtype=torch.float32, device=device)
    
    for start in range(0, N, batch):
        end = min(start + batch, N)
        w = weights[start:end]
        recons = a.unsqueeze(1) * c.unsqueeze(0)  # [K, C]
        errs = (w.unsqueeze(1).unsqueeze(2) - recons.unsqueeze(0)).abs()
        best = errs.view(end - start, -1).argmin(dim=1)
        atom_ids[start:end] = (best // C).to(torch.uint8)
        coeff_ids[start:end] = (best % C).to(torch.uint8)
        recon[start:end] = recons.view(-1)[best]
    
    # Residuals
    if residual_threshold > 0:
        residuals = weights - recon
        has_residual = residuals.abs() > residual_threshold
        residual_vals = residuals.to(torch.float16)
    else:
        has_residual = torch.zeros(N, dtype=torch.bool, device=device)
        residual_vals = torch.empty(0, dtype=torch.float16, device=device)
    
    prog = ProgramBufferV1(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residual_vals,
        has_residual=has_residual,
        shape=weights.shape,
    )
    return prog, recon


def build_hierarchical_atoms(base_atoms: torch.Tensor, programs: ProgramBufferV1,
                             max_l1: int = 64) -> AtomTableV1:
    """Build hierarchical atom definitions from program statistics.
    
    Analyzes which L0 atoms frequently appear in similar contexts
    and creates L1 composite atoms from common pairs.
    """
    K0 = base_atoms.numel()
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K0)]
    
    # Count co-occurrence of atom pairs in programs
    N = programs.N
    device = programs.atom_ids.device
    
    # Simple heuristic: find atoms that are frequently used together
    # Build pair counts from adjacent weights in the tensor
    flat_atoms = programs.atom_ids.reshape(-1)
    pairs = torch.stack([flat_atoms[:-1], flat_atoms[1:]], dim=1)  # [N-1, 2]
    
    # Count unique pairs
    # Use hash trick: pair_id = a1 * K0 + a2
    pair_ids = pairs[:, 0].long() * K0 + pairs[:, 1].long()
    counts = torch.bincount(pair_ids, minlength=K0 * K0)
    
    # Top frequent pairs become L1 atoms
    top_pairs = counts.argsort(descending=True)[:max_l1]
    
    for pair_idx in top_pairs:
        if counts[pair_idx] < 100:  # threshold
            break
        a1 = int(pair_idx // K0)
        a2 = int(pair_idx % K0)
        
        # L1 atom: ADD of two L0 atoms with equal weight
        l1_id = len(atom_defs)
        atom_defs.append(AtomDef(
            level=1,
            op="ADD",
            children=[a1, a2],
            coeffs=[0.5, 0.5],
        ))
    
    return AtomTableV1(base_atoms=base_atoms, atom_defs=atom_defs)
