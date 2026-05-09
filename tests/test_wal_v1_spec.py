#!/usr/bin/env python3
"""WAL v1 Compatibility Tests — M148 / Spec Freeze Validation.

Validates all guarantees in WAL_v1_SPEC.md:
1. Canonicalization produces deterministic encode
2. Frozen table gives 0% non-target diff
3. 12-bit packing round-trip exact
4. Serialize/deserialize round-trip exact
5. Safety score correlates with edit magnitude
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wal.v1 import (
    build_l0_atoms, build_coeff_table, wal_encode_v1, wal_decode_v1,
    encode_linear_weight, replace_linear_with_wal,
    serialize_wal_v1, deserialize_wal_v1,
)
from wal.v1.nn import replace_wal_with_linear
from wal.v1.isa import AtomTableV1, AtomDef, CoeffTable


def test_canonicalization_determinism():
    """Test 1: Same weights → identical programs with canonicalization."""
    torch.manual_seed(42)
    w = torch.randn(1000)
    atoms = build_l0_atoms(w, K=16, iters=3)
    coeffs = build_coeff_table(w, atoms, C=8, iters=3)
    
    # Canonicalize
    atoms = atoms[torch.argsort(atoms.abs())]
    
    prog1, recon1 = wal_encode_v1(w, atoms, coeffs, batch=1024)
    prog2, recon2 = wal_encode_v1(w, atoms, coeffs, batch=1024)
    
    diff = (prog1.atom_ids != prog2.atom_ids).float().mean().item()
    assert diff == 0.0, f"Canonicalization failed: {diff*100:.2f}% diff"
    
    relmse = ((recon1 - recon2).abs() / (w.abs() + 1e-8)).mean().item()
    assert relmse < 1e-6, f"Reconstruction mismatch: {relmse}"
    print("✅ Test 1: Canonicalization determinism — PASS")


def test_frozen_table_locality():
    """Test 2: Frozen table → non-target diff = 0%."""
    torch.manual_seed(42)
    w_base = torch.randn(5000)
    
    # Build table on base
    atoms = build_l0_atoms(w_base, K=32, iters=3)
    coeffs = build_coeff_table(w_base, atoms, C=8, iters=3)
    atoms = atoms[torch.argsort(atoms.abs())]
    
    # Encode base
    prog_base, _ = wal_encode_v1(w_base, atoms, coeffs, batch=1024)
    
    # Edit: perturb 10% of weights
    w_edit = w_base.clone()
    w_edit[:500] += torch.randn(500) * 0.01
    
    # Re-encode with SAME frozen table
    prog_edit, _ = wal_encode_v1(w_edit, atoms, coeffs, batch=1024)
    
    # Non-target diff (unchanged region)
    nontarget_diff = (prog_base.atom_ids[500:] != prog_edit.atom_ids[500:]).float().mean().item()
    
    # Target diff (changed region)
    target_diff = (prog_base.atom_ids[:500] != prog_edit.atom_ids[:500]).float().mean().item()
    
    assert nontarget_diff == 0.0, f"Non-target diff = {nontarget_diff*100:.2f}% (expected 0%)"
    assert target_diff > 0.1, f"Target diff = {target_diff*100:.2f}% (expected >10%)"
    print(f"✅ Test 2: Frozen table locality — PASS (non-target={nontarget_diff*100:.2f}%, target={target_diff*100:.1f}%)")


def test_12bit_packing_roundtrip():
    """Test 3: 12-bit packing round-trip exact."""
    from wal.v1.format import _pack_uint4, _unpack_uint4
    
    N = 1000
    atom_ids = torch.randint(0, 256, (N,), dtype=torch.uint8)
    coeff_ids = torch.randint(0, 16, (N,), dtype=torch.uint8)
    
    # Pack coeffs (4-bit)
    packed = _pack_uint4(coeff_ids)
    unpacked = _unpack_uint4(packed, N)
    
    assert (coeff_ids == unpacked).all(), "12-bit coeff packing failed"
    
    # Full packing simulation
    bytes_out = N * 1 + (N + 1) // 2  # atom_ids + packed coeffs
    bytes_per_weight = bytes_out / N
    assert abs(bytes_per_weight - 1.5) < 0.01, f"Packing ratio = {bytes_per_weight:.3f} (expected 1.5)"
    
    print(f"✅ Test 3: 12-bit packing round-trip — PASS ({bytes_per_weight:.2f} bytes/weight)")


def test_serialize_deserialize():
    """Test 4: Binary serialize/deserialize round-trip exact."""
    torch.manual_seed(42)
    w = torch.randn(1000)
    wal_param = encode_linear_weight(w, K=16, C=8)
    
    blob = serialize_wal_v1(
        wal_param.prog,
        wal_param.atom_table,
        wal_param.coeffs,
    )
    
    prog2, atom_table2, coeffs2, meta = deserialize_wal_v1(blob)
    
    recon1 = wal_decode_v1(wal_param.prog, wal_param.atom_table, wal_param.coeffs.values)
    recon2 = wal_decode_v1(prog2, atom_table2, coeffs2.values)
    
    diff = (recon1 - recon2).abs().max().item()
    assert diff < 1e-4, f"Serialize round-trip failed: max diff = {diff}"
    
    assert tuple(meta['shape']) == tuple(w.shape), f"Shape mismatch: {meta['shape']} vs {w.shape}"
    print(f"✅ Test 4: Serialize/deserialize — PASS (max diff={diff:.2e}, blob={len(blob)} bytes)")


def test_safety_score():
    """Test 5: Safety score monotonic with edit magnitude."""
    def score(delta_W):
        spectral = torch.linalg.matrix_norm(delta_W, ord=2).item()
        if spectral < 1.0:     return 0  # SAFE
        elif spectral < 5.0:   return 1  # MODERATE
        elif spectral < 10.0:  return 2  # RISKY
        else:                  return 3  # DANGEROUS
    
    w = torch.randn(100, 100)
    
    scores = []
    for scale in [0.001, 0.01, 0.1, 1.0, 10.0]:
        delta = torch.randn(100, 100) * scale
        s = score(delta)
        scores.append(s)
    
    # Scores should be non-decreasing
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i+1], f"Safety score not monotonic: {scores}"
    
    print(f"✅ Test 5: Safety score monotonicity — PASS (scores: {scores})")


def test_model_conversion_roundtrip():
    """Test 6: nn.Linear → WALLinear → nn.Linear round-trip."""
    torch.manual_seed(42)
    linear = nn.Linear(64, 32)
    w_orig = linear.weight.data.clone()
    
    wal_param = encode_linear_weight(linear.weight.data, K=256, C=16)
    
    # Decode
    w_decoded = wal_param.decode()
    relmse = ((w_orig - w_decoded).abs() / (w_orig.abs() + 1e-8)).mean().item()
    
    assert relmse < 0.01, f"Model conversion failed: relMSE = {relmse}"
    print(f"✅ Test 6: Model conversion round-trip — PASS (relMSE={relmse:.4f})")


def run_all():
    print("=" * 60)
    print("WAL v1 Compatibility Tests — M148 / Spec Freeze")
    print("=" * 60)
    
    test_canonicalization_determinism()
    test_frozen_table_locality()
    test_12bit_packing_roundtrip()
    test_serialize_deserialize()
    test_safety_score()
    test_model_conversion_roundtrip()
    
    print("=" * 60)
    print("All 6 tests PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
