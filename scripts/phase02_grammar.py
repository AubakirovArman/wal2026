#!/usr/bin/env python3
"""Phase 2 Demo: Text format round-trip."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.grammar import format_wal_text, parse_wal_text

print("=" * 60)
print("Phase 2: Grammar Round-trip")
print("=" * 60)

# Encode
weights = torch.randn(1024, 1024)
flat = weights.reshape(-1)
atoms = AtomTable(build_atoms_kmeans_v2(flat, K=64, iters=3))
coeffs = CoeffTable(build_coeff_table(flat, atoms.values, C=8, iters=3))
prog, _ = wal_encode_v2(flat, atoms, coeffs, shape=weights.shape)

# To text
text = format_wal_text(prog, atoms, coeffs)
print("Text format (first 10 lines):")
print("\n".join(text.splitlines()[:10]))
print("...")

# Back from text
prog2, atoms2, coeffs2 = parse_wal_text(text)

# Verify
match = torch.allclose(atoms.values, atoms2.values) and torch.allclose(coeffs.values, coeffs2.values)
print(f"  Round-trip exact: {'YES' if match else 'NO'}")
