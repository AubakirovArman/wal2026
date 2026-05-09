#!/usr/bin/env python3
"""Phase 4 Demo: Binary compression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.format import serialize_wal_v2, deserialize_wal_v2

print("=" * 60)
print("Phase 4: Binary Compression")
print("=" * 60)

# Encode
weights = torch.randn(4096, 4096)
flat = weights.reshape(-1)
atoms = AtomTable(build_atoms_kmeans_v2(flat, K=256, iters=3))
coeffs = CoeffTable(build_coeff_table(flat, atoms.values, C=16, iters=3))
prog, _ = wal_encode_v2(flat, atoms, coeffs, shape=weights.shape)
row_scales = torch.ones(weights.shape[0], dtype=torch.float32)

# Serialize
blob = serialize_wal_v2(prog, atoms, coeffs, row_scales)
print(f"  Serialized size: {len(blob):,} bytes")
print(f"  Dense size: {weights.numel() * 2:,} bytes (bf16)")
print(f"  Compression ratio: {weights.numel() * 2 / len(blob):.2f}x")

# Deserialize
prog2, atoms2, coeffs2, row_scales2, meta = deserialize_wal_v2(blob)
match = torch.allclose(atoms.values, atoms2.values) and torch.allclose(coeffs.values, coeffs2.values)
print(f"  Round-trip exact: {'YES' if match else 'NO'}")
