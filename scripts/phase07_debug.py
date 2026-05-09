#!/usr/bin/env python3
"""Phase 7 Demo: Debugger."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from wal.v1.isa import AtomTableV1, CoeffTable, ProgramBufferV1, AtomDef
from wal.v1.debugger import WALDebugger

print("=" * 60)
print("Phase 7: Debugger")
print("=" * 60)

# Create atom table
atoms = torch.randn(16, dtype=torch.float32)
atom_defs = [AtomDef(level=0, op="CONST") for _ in range(16)]
atom_table = AtomTableV1(atoms, atom_defs)

coeff_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
coeffs = CoeffTable(coeff_values)

# Create program
prog = ProgramBufferV1(
    atom_ids=torch.tensor([3, 7, 12, 5], dtype=torch.uint8),
    coeff_ids=torch.tensor([0, 1, 2, 3], dtype=torch.uint8),
    residuals=torch.empty(0, dtype=torch.float16),
    has_residual=torch.zeros(4, dtype=torch.bool),
    shape=(2, 2),
)

# Debug
dbg = WALDebugger(atom_table, coeffs)

# Step through
for i in range(4):
    record = dbg.step(prog, i)
    print(f"  Weight {i}: atom={record.atom_value:.4f} × coeff={record.coeff_value:.4f} = {record.final_value:.4f}")

# Heatmap
stats = dbg.heatmap(prog)
print(f"  Atom entropy: {stats.get('atom_entropy', 0):.4f}")
