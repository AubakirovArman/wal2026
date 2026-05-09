#!/usr/bin/env python3
"""M76: WAL v1 Round-trip Test.

Tests:
1. Binary serialization round-trip (encode → serialize → deserialize → decode)
2. Text format round-trip (text → assemble → disassemble)
3. Hierarchical atom resolution consistency (fast path vs interpretable path)
"""
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import numpy as np
from wal.v1 import (
    build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms,
    wal_decode_v1, precompute_flat_atoms, apply_row_scale,
    serialize_wal_v1, deserialize_wal_v1,
    assemble, disassemble, CoeffTable,
)


def test_binary_roundtrip():
    """Test binary serialization round-trip."""
    print("=" * 60)
    print("TEST 1: Binary Serialization Round-trip")
    print("=" * 60)
    
    # Generate test weights
    torch.manual_seed(42)
    weights = torch.randn(1024, dtype=torch.float32) * 0.1
    
    # Encode
    K, C = 16, 4
    atoms = build_l0_atoms(weights, K=K, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    
    # Build hierarchical atoms
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
    
    # Wrap coeffs in CoeffTable for serialization
    coeff_table = CoeffTable(values=coeffs)
    
    # Serialize
    blob = serialize_wal_v1(prog, atom_table, coeff_table, metadata={"test": "roundtrip", "layer": 0})
    size_kb = len(blob) / 1024
    
    # Deserialize
    prog2, atom_table2, coeffs2, meta = deserialize_wal_v1(blob)
    
    # Decode both (coeffs is a tensor from build_coeff_table)
    recon1 = wal_decode_v1(prog, atom_table, coeffs)
    recon2 = wal_decode_v1(prog2, atom_table2, coeffs2.values)
    
    # Compare
    max_diff = (recon1 - recon2).abs().max().item()
    mean_diff = (recon1 - recon2).abs().mean().item()
    
    # Also verify flat atoms consistency
    flat_atoms = precompute_flat_atoms(atom_table)
    flat_atoms2 = precompute_flat_atoms(atom_table2)
    flat_diff = (flat_atoms - flat_atoms2).abs().max().item()
    
    print(f"  Original shape: {weights.shape}")
    print(f"  K0={atom_table.K0}, K_total={atom_table.K_total}, C={C}, N={prog.N}")
    print(f"  Blob size: {size_kb:.2f} KB")
    print(f"  Metadata: {meta}")
    print(f"  Max reconstruction diff: {max_diff:.8f}")
    print(f"  Mean reconstruction diff: {mean_diff:.8f}")
    
    assert max_diff < 1e-5, f"Binary round-trip failed: max_diff={max_diff}"
    assert meta["test"] == "roundtrip", "Metadata mismatch"
    assert atom_table2.K0 == atom_table.K0, "K0 mismatch"
    assert atom_table2.K_total == atom_table.K_total, "K_total mismatch"
    assert flat_diff < 1e-5, f"Flat atoms mismatch: {flat_diff}"
    
    print("  ✅ PASS")
    return True


def test_text_roundtrip():
    """Test text format round-trip."""
    print()
    print("=" * 60)
    print("TEST 2: Text Format Round-trip")
    print("=" * 60)
    
    sample_text = """K 4
C 4
SHAPE 8

; Atom Definitions
ATOM 0 = 0.500000
ATOM 1 = -0.250000
ATOM 2 = 0.125000
ATOM 3 = -0.125000
ATOM 4 = ADD(ATOM 0 * 0.500000, ATOM 1 * 0.500000)
ATOM 5 = MUL(ATOM 2 * 1.000000, ATOM 3 * 1.000000)

; Programs
ATOM 0 COEF 0.500000
ATOM 1 COEF 1.000000
ATOM 4 COEF 0.750000
ATOM 2 COEF 0.250000
ATOM 5 COEF 1.500000
ATOM 3 COEF 0.800000
ATOM 0 COEF 1.200000
ATOM 1 COEF 0.300000
"""
    
    # Assemble
    prog, atom_table, coeffs, meta = assemble(sample_text)
    
    print(f"  Assembled: K0={atom_table.K0}, K_total={atom_table.K_total}, N={prog.N}")
    print(f"  Hierarchical atoms: {atom_table.K_total - atom_table.K0}")
    
    # Disassemble
    text2 = disassemble(prog, atom_table, coeffs, include_defs=True)
    
    # Re-assemble
    prog2, atom_table2, coeffs2, meta2 = assemble(text2)
    
    # Compare reconstructions
    recon1 = wal_decode_v1(prog, atom_table, coeffs.values)
    recon2 = wal_decode_v1(prog2, atom_table2, coeffs2.values)
    
    max_diff = (recon1 - recon2).abs().max().item()
    
    print(f"  Max reconstruction diff after text round-trip: {max_diff:.8f}")
    
    assert max_diff < 1e-5, f"Text round-trip failed: max_diff={max_diff}"
    
    print("  ✅ PASS")
    return True


def test_hierarchical_consistency():
    """Test that fast path and interpretable path give same results."""
    print()
    print("=" * 60)
    print("TEST 3: Hierarchical Path Consistency")
    print("=" * 60)
    
    torch.manual_seed(123)
    weights = torch.randn(512, dtype=torch.float32) * 0.05
    
    K, C = 32, 8
    atoms = build_l0_atoms(weights, K=K, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=16)
    
    # Fast path (precomputed flat atoms)
    recon_fast = wal_decode_v1(prog, atom_table, coeffs, use_hierarchical=False)
    
    # Interpretable path (recursive resolve)
    recon_hier = wal_decode_v1(prog, atom_table, coeffs, use_hierarchical=True)
    
    max_diff = (recon_fast - recon_hier).abs().max().item()
    mean_diff = (recon_fast - recon_hier).abs().mean().item()
    
    print(f"  K0={atom_table.K0}, K_total={atom_table.K_total}")
    print(f"  Fast path max diff vs interpretable: {max_diff:.8f}")
    print(f"  Fast path mean diff vs interpretable: {mean_diff:.8f}")
    
    assert max_diff < 1e-4, f"Hierarchical consistency failed: max_diff={max_diff}"
    
    print("  ✅ PASS")
    return True


def test_binary_with_hierarchical_atoms():
    """Test binary round-trip with hierarchical atoms."""
    print()
    print("=" * 60)
    print("TEST 4: Binary Round-trip with Hierarchical Atoms")
    print("=" * 60)
    
    torch.manual_seed(777)
    weights = torch.randn(2048, dtype=torch.float32) * 0.08
    
    K, C = 64, 16
    atoms = build_l0_atoms(weights, K=K, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    
    # Build many L1 atoms
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=32)
    
    print(f"  Created {atom_table.K_total - atom_table.K0} L1 atoms")
    
    # Wrap coeffs for serialization
    coeff_table = CoeffTable(values=coeffs)
    
    # Binary round-trip
    blob = serialize_wal_v1(prog, atom_table, coeff_table)
    prog2, atom_table2, coeffs2, _ = deserialize_wal_v1(blob)
    
    # Check atom definitions
    for i in range(atom_table.K0, atom_table.K_total):
        d1 = atom_table.atom_defs[i]
        d2 = atom_table2.atom_defs[i]
        assert d1.op == d2.op, f"Op mismatch at {i}: {d1.op} vs {d2.op}"
        assert d1.children == d2.children, f"Children mismatch at {i}"
        assert d1.coeffs == d2.coeffs, f"Coeffs mismatch at {i}"
    
    # Check reconstructions match
    recon1_fast = wal_decode_v1(prog, atom_table, coeffs, use_hierarchical=False)
    recon2_fast = wal_decode_v1(prog2, atom_table2, coeffs2.values, use_hierarchical=False)
    recon2_hier = wal_decode_v1(prog2, atom_table2, coeffs2.values, use_hierarchical=True)
    
    diff_fast = (recon1_fast - recon2_fast).abs().max().item()
    diff_hier = (recon1_fast - recon2_hier).abs().max().item()
    
    print(f"  Fast path round-trip diff: {diff_fast:.8f}")
    print(f"  Interpretable path round-trip diff: {diff_hier:.8f}")
    
    assert diff_fast < 1e-5, f"Fast round-trip failed: {diff_fast}"
    assert diff_hier < 1e-4, f"Hier round-trip failed: {diff_hier}"
    
    print("  ✅ PASS")
    return True


def test_text_binary_combined_roundtrip():
    """Test: text → assemble → binary → deserialize → disassemble → text."""
    print()
    print("=" * 60)
    print("TEST 5: Text → Binary → Text Combined Round-trip")
    print("=" * 60)
    
    sample_text = """K 4
C 4
SHAPE 16

; Atom Definitions
ATOM 0 = 1.000000
ATOM 1 = -1.000000
ATOM 2 = 0.500000
ATOM 3 = -0.500000
ATOM 4 = ADD(ATOM 0 * 0.700000, ATOM 1 * 0.300000)
ATOM 5 = ADD(ATOM 2 * 0.500000, ATOM 3 * 0.500000)

; Programs
ATOM 0 COEF 1.000000
ATOM 1 COEF 0.500000
ATOM 4 COEF 0.800000
ATOM 2 COEF 0.250000
ATOM 5 COEF 1.000000
ATOM 3 COEF 0.600000
ATOM 0 COEF 0.400000
ATOM 1 COEF 1.200000
ATOM 4 COEF 0.900000
ATOM 2 COEF 0.300000
ATOM 5 COEF 0.700000
ATOM 3 COEF 0.500000
ATOM 0 COEF 0.200000
ATOM 1 COEF 0.100000
ATOM 4 COEF 1.000000
ATOM 2 COEF 0.800000
"""
    
    # Text → assemble
    prog, atom_table, coeffs, meta = assemble(sample_text)
    
    # → binary
    blob = serialize_wal_v1(prog, atom_table, coeffs)
    
    # → deserialize
    prog2, atom_table2, coeffs2, _ = deserialize_wal_v1(blob)
    
    # → disassemble
    text2 = disassemble(prog2, atom_table2, coeffs2, include_defs=True)
    
    # Re-assemble text2 to verify
    prog3, atom_table3, coeffs3, _ = assemble(text2)
    
    recon1 = wal_decode_v1(prog, atom_table, coeffs.values)
    recon3 = wal_decode_v1(prog3, atom_table3, coeffs3.values)
    
    max_diff = (recon1 - recon3).abs().max().item()
    
    print(f"  Text→Binary→Text round-trip diff: {max_diff:.8f}")
    print(f"  Blob size: {len(blob)} bytes")
    
    assert max_diff < 1e-5, f"Combined round-trip failed: {max_diff}"
    
    print("  ✅ PASS")
    return True


def main():
    print("\n" + "=" * 60)
    print("M76: WAL v1 Round-trip Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Binary Round-trip", test_binary_roundtrip()))
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results.append(("Binary Round-trip", False))
    
    try:
        results.append(("Text Round-trip", test_text_roundtrip()))
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results.append(("Text Round-trip", False))
    
    try:
        results.append(("Hierarchical Consistency", test_hierarchical_consistency()))
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results.append(("Hierarchical Consistency", False))
    
    try:
        results.append(("Binary + Hierarchical", test_binary_with_hierarchical_atoms()))
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results.append(("Binary + Hierarchical", False))
    
    try:
        results.append(("Text→Binary→Text", test_text_binary_combined_roundtrip()))
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results.append(("Text→Binary→Text", False))
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
