"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M81: Phase 10 — Meta-Learning Tests.

Test WALProgramAdapter, WALCoeffAdapter, program_soup, evolve_programs.
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import torch.nn as nn
from wal.v1.isa import AtomTableV1, CoeffTable, ProgramBufferV1, AtomDef
from wal.v1.meta import (
    WALProgramAdapter, WALCoeffAdapter, WALAtomAdapter,
    program_soup, evolve_programs, compute_program_gradient,
)

print("=" * 60)
print("M81: Phase 10 — Meta-Learning")
print("=" * 60)

# ---- Setup ----
torch.manual_seed(81)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create atom table and coeffs
base_atoms = torch.randn(32, dtype=torch.float32, device=device)

# Build atom_defs: 32 L0 + 8 L1
atom_defs = []
for i in range(32):
    atom_defs.append(AtomDef(level=0, op="CONST"))
for i in range(32, 40):
    atom_defs.append(AtomDef(
        level=1,
        op="ADD",
        children=[i % 32, (i + 1) % 32],
        coeffs=[0.5, 0.5],
    ))

atom_table = AtomTableV1(base_atoms, atom_defs)

coeff_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32, device=device)
coeffs = CoeffTable(coeff_values)

# Create a simple program
shape = (10, 10)
N = 100
prog = ProgramBufferV1(
    atom_ids=torch.randint(0, 40, (N,), dtype=torch.uint8, device=device),
    coeff_ids=torch.randint(0, 4, (N,), dtype=torch.uint8, device=device),
    residuals=torch.empty(0, dtype=torch.float16, device=device),
    has_residual=torch.zeros(N, dtype=torch.bool, device=device),
    shape=shape,
)

# ---- Test 1: WALProgramAdapter ----
print("\n[1/5] WALProgramAdapter")
base_weight = torch.randn(10, 10, device=device)
adapter = WALProgramAdapter(shape=(10, 10), rank=4, alpha=2.0).to(device)

# Set non-zero weights for testing
with torch.no_grad():
    adapter.lora_A.normal_(0, 0.1)
    adapter.lora_B.normal_(0, 0.1)

# Forward
adapted = adapter(base_weight)
assert adapted.shape == (10, 10), "Shape mismatch"
assert not torch.equal(adapted, base_weight), "Adapter had no effect"

# Merge
merged = adapter.merge(base_weight)
assert torch.allclose(adapted, merged), "Merge != forward"

# Parameter count
param_count = sum(p.numel() for p in adapter.parameters())
assert param_count == 10 * 4 + 4 * 10, f"Expected 140 params, got {param_count}"

print(f"  ✓ ProgramAdapter: rank=4, params={param_count}, shape={adapted.shape}")

# ---- Test 2: WALCoeffAdapter ----
print("\n[2/5] WALCoeffAdapter")
coeff_adapter = WALCoeffAdapter(num_coeffs=4, init_scale=0.01).to(device)

base_coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
adapted_coeffs = coeff_adapter.adapt_coeffs(base_coeffs)

assert adapted_coeffs.shape == (4,), "Shape mismatch"
assert not torch.equal(adapted_coeffs, base_coeffs), "No adaptation"
assert torch.allclose(adapted_coeffs - base_coeffs, coeff_adapter.coeff_delta, atol=1e-5), "Delta mismatch"

param_count = sum(p.numel() for p in coeff_adapter.parameters())
assert param_count == 4, f"Expected 4 params, got {param_count}"

print(f"  ✓ CoeffAdapter: num_coeffs=4, params={param_count}")

# ---- Test 3: WALAtomAdapter ----
print("\n[3/5] WALAtomAdapter")
atom_adapter = WALAtomAdapter(num_atoms=40, num_adapt=8, init_scale=0.01).to(device)

base_atoms_tensor = torch.randn(40, device=device)
adapted_atoms = atom_adapter.adapt_atoms(base_atoms_tensor)

assert adapted_atoms.shape == (40,), "Shape mismatch"
assert not torch.equal(adapted_atoms, base_atoms_tensor), "No adaptation"

# Only first 8 should be adapted
for i in range(8):
    assert not torch.isclose(adapted_atoms[i], base_atoms_tensor[i]), f"Atom {i} not adapted"
for i in range(8, 40):
    assert torch.isclose(adapted_atoms[i], base_atoms_tensor[i]), f"Atom {i} unexpectedly adapted"

param_count = sum(p.numel() for p in atom_adapter.parameters())
assert param_count == 40, f"Expected 40 params, got {param_count}"

print(f"  ✓ AtomAdapter: num_atoms=40, num_adapt=8, params={param_count}")

# ---- Test 4: Program Soup ----
print("\n[4/5] Program Soup")
progs = []
for i in range(3):
    p = ProgramBufferV1(
        atom_ids=torch.randint(0, 40, (N,), dtype=torch.uint8, device=device),
        coeff_ids=torch.randint(0, 4, (N,), dtype=torch.uint8, device=device),
        residuals=torch.empty(0, dtype=torch.float16, device=device),
        has_residual=torch.zeros(N, dtype=torch.bool, device=device),
        shape=shape,
    )
    progs.append(p)

# Test mean method
soup_mean = program_soup(progs, method="mean")
assert soup_mean.N == N, "N mismatch"
assert soup_mean.shape == shape, "Shape mismatch"

# Test majority method
soup_majority = program_soup(progs, method="majority")
assert soup_majority.N == N, "N mismatch"

# Test weighted
weights = [0.5, 0.3, 0.2]
soup_weighted = program_soup(progs, weights=weights, method="weighted")
assert soup_weighted.N == N, "N mismatch"

print(f"  ✓ Program soup: 3 programs → mean/majority/weighted")

# ---- Test 5: Evolution ----
print("\n[5/5] Genetic Evolution")
target_weights = torch.randn(20, 5, device=device)

best_prog, recon = evolve_programs(
    target_weights,
    atom_table,
    coeffs,
    population_size=8,
    generations=5,
    mutation_rate=0.1,
    top_k=2,
)

mse = (target_weights - recon).pow(2).mean().item()
assert best_prog.N == 100, f"N mismatch: {best_prog.N}"
assert recon.shape == target_weights.shape, f"Shape mismatch: {recon.shape}"
assert mse < 10.0, f"MSE too high: {mse:.6f}"

print(f"  ✓ Evolution: 8 pop × 5 gen → MSE={mse:.6f}")

# ---- Summary ----
print("\n" + "=" * 60)
print("M81: ALL 5/5 TESTS PASS")
print("=" * 60)
print("\nPhase 10 components:")
print("  • WALProgramAdapter — LoRA-style residual adapter")
print("  • WALCoeffAdapter — learned coefficient offsets")
print("  • WALAtomAdapter — learned atom perturbations")
print("  • program_soup — merge programs from N models")
print("  • evolve_programs — genetic algorithm on atom combinations")
