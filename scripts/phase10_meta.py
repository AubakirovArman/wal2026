#!/usr/bin/env python3
"""Phase 10 Demo: Meta-learning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v1.meta import WALProgramAdapter, WALCoeffAdapter, program_soup
from wal.v1.isa import ProgramBufferV1

print("=" * 60)
print("Phase 10: Meta-Learning")
print("=" * 60)

# ProgramAdapter
adapter = WALProgramAdapter(shape=(512, 1024), rank=4, alpha=1.0)
base_weight = torch.randn(512, 1024)
adapted = adapter(base_weight)
params = sum(p.numel() for p in adapter.parameters())
print(f"  ProgramAdapter: rank=4, params={params}")

# CoeffAdapter
coeff_adapter = WALCoeffAdapter(num_coeffs=16)
base_coeffs = torch.randn(16)
adapted_coeffs = coeff_adapter.adapt_coeffs(base_coeffs)
print(f"  CoeffAdapter: num_coeffs=16, params={sum(p.numel() for p in coeff_adapter.parameters())}")

# Program soup
progs = []
for _ in range(3):
    prog = ProgramBufferV1(
        atom_ids=torch.randint(0, 16, (100,), dtype=torch.uint8),
        coeff_ids=torch.randint(0, 4, (100,), dtype=torch.uint8),
        residuals=torch.empty(0, dtype=torch.float16),
        has_residual=torch.zeros(100, dtype=torch.bool),
        shape=(10, 10),
    )
    progs.append(prog)

soup = program_soup(progs, method="mean")
print(f"  Program soup: 3 programs → 1 merged program ({soup.N} weights)")
