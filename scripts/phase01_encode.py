#!/usr/bin/env python3
"""Phase 1 Demo: Encode a tensor to WAL format."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1

print("=" * 60)
print("Phase 1: Encode Tensor to WAL")
print("=" * 60)

# Create sample weights (smaller for demo)
weights = torch.randn(1024, 1024)
flat = weights.reshape(-1)

# Encode
print(f"Encoding {weights.numel()} weights...")
atoms = build_l0_atoms(flat, K=64, iters=3)
coeffs = build_coeff_table(flat, atoms, C=8, iters=3)
prog, recon = wal_encode_v1(flat, atoms, coeffs)

# Stats
mse = (flat - recon).pow(2).mean().item()
rel_mse = mse / flat.pow(2).mean().item()
bpw = 12  # 8 bits atom_id + 4 bits coeff_id

print(f"  MSE: {mse:.8f}")
print(f"  relMSE: {rel_mse:.8f}")
print(f"  Bits/weight: {bpw}")
print(f"  Unique programs: {prog.atom_ids.numel()}")
print("  Done.")
