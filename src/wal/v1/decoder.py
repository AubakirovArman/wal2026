#!/usr/bin/env python3
"""WAL v1 decoder — supports hierarchical atom resolution.

Fast path: precompute flattened atom values (same as WAL v2).
Interpretable path: recursively resolve hierarchical definitions.
"""
import torch
from .isa import AtomTableV1, ProgramBufferV1, CoeffTable


def precompute_flat_atoms(atom_table: AtomTableV1) -> torch.Tensor:
    """Precompute flattened scalar values for all atoms (fast decode)."""
    return torch.tensor(
        [atom_table.resolve(i) for i in range(atom_table.K_total)],
        dtype=torch.float32,
        device=atom_table.base_atoms.device,
    )


def wal_decode_v1(prog: ProgramBufferV1, atom_table: AtomTableV1, 
                  coeff_values: torch.Tensor, use_hierarchical: bool = False) -> torch.Tensor:
    """Decode WAL v1 programs.
    
    Args:
        prog: ProgramBufferV1
        atom_table: Hierarchical atom table
        coeff_values: Coefficient values tensor [C]
        use_hierarchical: If True, recursively resolve atom definitions (slow but interpretable).
                         If False, use precomputed flattened values (fast, same as v2).
    """
    device = prog.atom_ids.device
    N = prog.N
    
    if use_hierarchical:
        # Slow path: recursively resolve each atom
        recon = torch.empty(N, dtype=torch.float32, device=device)
        for i in range(N):
            atom_val = atom_table.resolve(int(prog.atom_ids[i]))
            coeff_val = coeff_values[int(prog.coeff_ids[i])].item()
            recon[i] = atom_val * coeff_val
    else:
        # Fast path: precompute all atom values (includes L1+ atoms)
        flat_atoms = precompute_flat_atoms(atom_table).to(device)
        recon = flat_atoms[prog.atom_ids.long()] * coeff_values[prog.coeff_ids.long()]
    
    # Add residuals
    if prog.residuals.numel() > 0:
        recon = recon + prog.residuals.float()
    
    return recon.reshape(prog.shape)


def apply_row_scale(recon: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    """Denormalize by row scales."""
    return recon * row_scale
