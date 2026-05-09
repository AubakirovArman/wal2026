#!/usr/bin/env python3
"""Phase 3 Demo: Decode benchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import time
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.decoder import wal_decode_v1
from wal.v1.isa import AtomTableV1, AtomDef

print("=" * 60)
print("Phase 3: Decode Benchmark")
print("=" * 60)

device = torch.device("cpu")  # CPU for stable demo
print(f"Device: {device}")

# Encode (smaller for demo, on CPU to avoid OOM)
weights = torch.randn(2048, 2048)
flat = weights.reshape(-1)
atoms_tensor = build_l0_atoms(flat, K=128, iters=3)
coeffs = build_coeff_table(flat, atoms_tensor, C=8, iters=3)
prog, _ = wal_encode_v1(flat, atoms_tensor, coeffs)

# Build AtomTableV1 for decoder
atom_defs = [AtomDef(level=0, op="CONST") for _ in range(atoms_tensor.numel())]
atoms = AtomTableV1(atoms_tensor, atom_defs)

# Move to target device for benchmark
prog.atom_ids = prog.atom_ids.to(device)
prog.coeff_ids = prog.coeff_ids.to(device)
atoms.base_atoms = atoms.base_atoms.to(device)
coeffs = coeffs.to(device) if hasattr(coeffs, 'to') else coeffs

# Benchmark decode
warmup = 5
iterations = 20

for _ in range(warmup):
    _ = wal_decode_v1(prog, atoms, coeffs)

torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()
for _ in range(iterations):
    decoded = wal_decode_v1(prog, atoms, coeffs)
torch.cuda.synchronize() if device.type == "cuda" else None
elapsed = time.time() - start

total = weights.numel() * iterations
speed = total / elapsed / 1e6
print(f"  Decoded {total:,} weights in {elapsed:.3f}s")
print(f"  Speed: {speed:.1f} Mw/s")
