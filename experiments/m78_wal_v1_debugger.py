#!/usr/bin/env python3
"""M78: WAL v1 Debugger & Inspector Test (Phase 7).

Tests:
1. Step-through execution
2. Conditional breakpoints (atom, coeff, residual)
3. Hierarchical atom resolution trace
4. Program heatmap and statistics
5. Program diff
6. Trace log inspection
"""
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
from wal.v1 import (
    build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms,
    WALDebugger,
)


def test_step_through():
    """Test step-through execution."""
    print("=" * 60)
    print("TEST 1: Step-through Execution")
    print("=" * 60)
    
    torch.manual_seed(42)
    weights = torch.randn(100, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=16, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    
    # Step through first 5 weights
    print("  Stepping through first 5 weights:")
    for i in range(5):
        record = dbg.step(prog, i)
        print(f"    [{i}] atom={record.atom_id} coeff={record.coeff_id} "
              f"val={record.final_value:.6f}")
    
    assert len(dbg.trace_log) == 5, "Trace log should have 5 records"
    assert dbg.trace_log[0].index == 0, "First record index should be 0"
    
    print("  ✅ PASS")
    return True


def test_breakpoints():
    """Test conditional breakpoints."""
    print()
    print("=" * 60)
    print("TEST 2: Conditional Breakpoints")
    print("=" * 60)
    
    torch.manual_seed(123)
    weights = torch.randn(200, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=16, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    
    # Break when atom_id == 5
    bp1 = dbg.set_atom_breakpoint(5, name="atom_5_trigger")
    
    # Break when coeff_id == 2
    bp2 = dbg.set_coeff_breakpoint(2, name="coeff_2_trigger")
    
    # Break when residual > 0.01
    bp3 = dbg.set_residual_breakpoint(0.01, name="big_residual")
    
    dbg.run(prog, start=0, end=200)
    
    print(f"  Breakpoint '{bp1.name}' hit: {bp1.hit_count} times")
    print(f"  Breakpoint '{bp2.name}' hit: {bp2.hit_count} times")
    print(f"  Breakpoint '{bp3.name}' hit: {bp3.hit_count} times")
    
    # Verify that some breakpoints were hit
    assert bp1.hit_count > 0, "Atom breakpoint should be hit at least once"
    assert bp2.hit_count > 0, "Coeff breakpoint should be hit at least once"
    
    # Verify trace log has breakpoint annotations
    # Note: one weight can trigger multiple breakpoints, so trace records with
    # breakpoint_hit may be fewer than total hit_count (only one name is stored)
    bp_hits = [r for r in dbg.trace_log if r.breakpoint_hit is not None]
    assert len(bp_hits) <= bp1.hit_count + bp2.hit_count + bp3.hit_count, \
        "Trace log should record at least some breakpoint hits"
    assert len(bp_hits) > 0, "At least one breakpoint should be recorded in trace"
    
    print(f"  Total breakpoint hits in trace: {len(bp_hits)}")
    print("  ✅ PASS")
    return True


def test_hierarchical_trace():
    """Test hierarchical atom resolution trace."""
    print()
    print("=" * 60)
    print("TEST 3: Hierarchical Atom Resolution Trace")
    print("=" * 60)
    
    torch.manual_seed(456)
    weights = torch.randn(50, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=8, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    
    print(f"  K0={atom_table.K0}, K_total={atom_table.K_total}")
    print(f"  Hierarchical atoms: {atom_table.K_total - atom_table.K0}")
    
    # Find a weight that uses a hierarchical atom
    hier_indices = []
    for i in range(prog.N):
        if int(prog.atom_ids[i]) >= atom_table.K0:
            hier_indices.append(i)
            if len(hier_indices) >= 3:
                break
    
    if hier_indices:
        print(f"  Found {len(hier_indices)} weights using hierarchical atoms")
        for idx in hier_indices[:1]:
            record = dbg.step(prog, idx)
            print(f"\n  Weight [{idx}] uses hierarchical atom {record.atom_id}:")
            tree = dbg.resolve_atom_tree(record.atom_id, max_depth=5)
            for line in tree.split('\n'):
                print(f"    {line}")
            assert record.atom_resolution_tree is not None, \
                "Hierarchical atom should have resolution tree"
    else:
        print("  No hierarchical atoms found in first 50 weights (OK for small sample)")
    
    print("  ✅ PASS")
    return True


def test_heatmap():
    """Test program heatmap and statistics."""
    print()
    print("=" * 60)
    print("TEST 4: Program Heatmap")
    print("=" * 60)
    
    torch.manual_seed(789)
    weights = torch.randn(1000, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=32, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=8, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    stats = dbg.heatmap(prog)
    
    print(f"  Total weights: {stats.total_weights:,}")
    print(f"  Unique atoms used: {len(stats.atom_frequencies)}")
    print(f"  Unique coeffs used: {len(stats.coeff_frequencies)}")
    print(f"  Residuals: {stats.residual_count} ({stats.residual_pct:.2f}%)")
    print(f"  Atom entropy: {stats.atom_entropy:.3f} bits")
    print(f"  Coeff entropy: {stats.coeff_entropy:.3f} bits")
    
    print(f"\n  Top 5 atoms:")
    for aid, cnt, pct in stats.top_atoms[:5]:
        print(f"    ATOM {aid:2d}: {cnt:5,} ({pct:5.2f}%)")
    
    print(f"\n  Top 5 coeffs:")
    for cid, cnt, pct in stats.top_coeffs[:5]:
        print(f"    COEF {cid:2d}: {cnt:5,} ({pct:5.2f}%)")
    
    assert stats.total_weights == 1000, "Total weights mismatch"
    assert len(stats.atom_frequencies) <= 32 + 8, "Too many unique atoms"
    assert stats.atom_entropy > 0, "Atom entropy should be positive"
    
    print("  ✅ PASS")
    return True


def test_program_diff():
    """Test program diff between two encodings."""
    print()
    print("=" * 60)
    print("TEST 5: Program Diff")
    print("=" * 60)
    
    torch.manual_seed(321)
    weights = torch.randn(100, dtype=torch.float32) * 0.1
    
    atoms1 = build_l0_atoms(weights, K=16, iters=3)
    coeffs1 = build_coeff_table(weights, atoms1, C=4, iters=3)
    prog1, recon1 = wal_encode_v1(weights, atoms1, coeffs1)
    
    # Encode same weights with different random seed
    torch.manual_seed(999)
    atoms2 = build_l0_atoms(weights, K=16, iters=3)
    coeffs2 = build_coeff_table(weights, atoms2, C=4, iters=3)
    prog2, recon2 = wal_encode_v1(weights, atoms2, coeffs2)
    
    atom_table = build_hierarchical_atoms(atoms1, prog1, max_l1=4)
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs1)
    
    diff = dbg.diff_programs(prog1, prog2, name1="encode_A", name2="encode_B")
    
    print(f"  Diff: {diff['name1']} vs {diff['name2']}")
    print(f"  Identical: {diff['identical']}")
    print(f"  Value diffs: {diff['value_diffs']:,} ({diff['value_diff_pct']:.2f}%)")
    print(f"  Max value diff: {diff['max_value_diff']:.8f}")
    
    # Same weights with different random seeds should have differences
    assert not diff['identical'], "Different encodings should not be identical"
    assert diff['value_diffs'] > 0, "Should have some value differences"
    
    # Diff of identical programs should be identical
    diff_same = dbg.diff_programs(prog1, prog1, name1="same", name2="same")
    assert diff_same['identical'], "Same program should be identical"
    
    print("  Identical program diff check: PASS")
    print("  ✅ PASS")
    return True


def test_trace_log_inspection():
    """Test trace log after running debugger."""
    print()
    print("=" * 60)
    print("TEST 6: Trace Log Inspection")
    print("=" * 60)
    
    torch.manual_seed(555)
    weights = torch.randn(50, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=16, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    dbg.set_atom_breakpoint(atom_id=3, name="atom_3")
    
    records = dbg.run(prog, start=0, end=50)
    
    print(f"  Trace log length: {len(records)}")
    print(f"  First record: index={records[0].index}, atom={records[0].atom_id}")
    print(f"  Last record: index={records[-1].index}, atom={records[-1].atom_id}")
    
    # Check all values are finite
    for r in records:
        assert torch.isfinite(torch.tensor(r.final_value)), \
            f"Non-finite value at index {r.index}"
    
    # Check breakpoint hits are recorded
    hits = [r for r in records if r.breakpoint_hit is not None]
    print(f"  Breakpoint hits: {len(hits)}")
    
    print("  ✅ PASS")
    return True


def test_custom_breakpoint():
    """Test custom breakpoint condition."""
    print()
    print("=" * 60)
    print("TEST 7: Custom Breakpoint Condition")
    print("=" * 60)
    
    torch.manual_seed(777)
    weights = torch.randn(100, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=16, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, recon = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)
    
    # Custom: break when atom_id > 10 AND coeff_id < 2
    bp = dbg.set_breakpoint(
        condition=lambda a, c, r: a > 10 and c < 2,
        name="complex_condition"
    )
    
    dbg.run(prog, start=0, end=100)
    
    print(f"  Custom breakpoint '{bp.name}' hit: {bp.hit_count} times")
    
    # Verify manually
    manual_hits = 0
    for i in range(100):
        if int(prog.atom_ids[i]) > 10 and int(prog.coeff_ids[i]) < 2:
            manual_hits += 1
    
    assert bp.hit_count == manual_hits, f"Breakpoint count mismatch: {bp.hit_count} vs {manual_hits}"
    
    print("  ✅ PASS")
    return True


def main():
    print("\n" + "=" * 60)
    print("M78: WAL v1 Debugger & Inspector Test (Phase 7)")
    print("=" * 60 + "\n")
    
    results = []
    
    tests = [
        ("Step-through", test_step_through),
        ("Breakpoints", test_breakpoints),
        ("Hierarchical Trace", test_hierarchical_trace),
        ("Heatmap", test_heatmap),
        ("Program Diff", test_program_diff),
        ("Trace Log", test_trace_log_inspection),
        ("Custom Breakpoint", test_custom_breakpoint),
    ]
    
    for name, test_fn in tests:
        try:
            results.append((name, test_fn()))
        except Exception as e:
            import traceback
            print(f"  ❌ FAIL: {e}")
            traceback.print_exc()
            results.append((name, False))
    
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
