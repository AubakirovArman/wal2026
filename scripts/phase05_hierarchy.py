#!/usr/bin/env python3
"""Phase 5 Demo: Hierarchical atoms."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v1.isa import AtomTableV1, AtomDef
from wal.v1.encoder import build_hierarchical_atoms
from wal.v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2

print("=" * 60)
print("Phase 5: Hierarchical Atoms")
print("=" * 60)

# Encode flat
weights = torch.randn(4096, 4096)
flat = weights.reshape(-1)
atoms = build_atoms_kmeans_v2(flat, K=16, iters=3)
coeffs = build_coeff_table(flat, atoms, C=8, iters=3)
prog, _ = wal_encode_v2(flat, atoms, coeffs)

# Build hierarchy
atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
print(f"  L0 atoms: {atom_table.K0}")
print(f"  L1 atoms: {atom_table.K_total - atom_table.K0}")
print(f"  Total atoms: {atom_table.K_total}")

# Resolve a composite
if atom_table.K_total > atom_table.K0:
    val = atom_table.resolve(atom_table.K0)
    print(f"  L1 atom #{atom_table.K0} value: {val:.6f}")
