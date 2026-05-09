"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M79: WAL v1 Standard Library Prototype (Phase 8).

Tests:
1. Build library entry from encoded weights
2. Save/load library to disk
3. Transfer atoms from source to target distribution
4. Evaluate transfer quality vs baseline
5. Direct transfer (scale-only, no fine-tuning)
"""
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import tempfile
import shutil
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.stdlib import (
    AtomLibrary, AtomLibraryEntry,
    build_entry_from_encoded,
    encode_with_pretrained_atoms,
    evaluate_transfer,
    transfer_atoms_direct,
    create_default_library,
)


def test_build_entry():
    """Test building library entry from encoded weights."""
    print("=" * 60)
    print("TEST 1: Build Library Entry")
    print("=" * 60)
    
    torch.manual_seed(42)
    weights = torch.randn(1024, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=32, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=8, iters=3)
    
    entry = build_entry_from_encoded(
        name="llama-70b-attention-l0",
        family="llama",
        variant="70b",
        component="attention",
        atoms=atoms,
        coeffs=coeffs,
        metadata={"layer": 0, "param": "q_proj"},
    )
    
    print(f"  Entry: {entry.name}")
    print(f"  Family: {entry.family}, Variant: {entry.variant}")
    print(f"  K={entry.K}, C={entry.C}")
    print(f"  Atom tensor shape: {entry.atom_tensor.shape}")
    print(f"  Coeff tensor shape: {entry.coeff_tensor.shape}")
    print(f"  Metadata: {entry.metadata}")
    
    assert entry.K == 32
    assert entry.C == 8
    assert entry.atom_tensor.numel() == 32
    assert entry.coeff_tensor.numel() == 8
    
    print("  ✅ PASS")
    return True


def test_library_save_load():
    """Test save/load library to disk."""
    print()
    print("=" * 60)
    print("TEST 2: Library Save/Load")
    print("=" * 60)
    
    # Build a small library
    lib = create_default_library()
    
    for name, family, variant in [
        ("llama-70b-all", "llama", "70b"),
        ("llama-8b-all", "llama", "8b"),
        ("mistral-7b-all", "mistral", "7b"),
    ]:
        atoms = torch.randn(64, dtype=torch.float32)
        coeffs = torch.linspace(0.5, 2.0, 16, dtype=torch.float32)
        entry = build_entry_from_encoded(
            name=name, family=family, variant=variant,
            component="all", atoms=atoms, coeffs=coeffs,
        )
        lib.add_entry(entry)
    
    print(f"  Created library: {lib}")
    print(f"  Families: {lib.families()}")
    print(f"  Variants (llama): {lib.variants('llama')}")
    print(f"  Entries: {lib.list_entries()}")
    
    # Save to temp dir
    tmpdir = tempfile.mkdtemp()
    try:
        lib.save(tmpdir)
        
        # Load back
        lib2 = AtomLibrary.load(tmpdir)
        
        print(f"  Loaded library: {lib2}")
        assert len(lib2.entries) == 3
        assert lib2.families() == ["llama", "mistral"]
        
        # Check entry
        entry = lib2.get_entry("llama-70b-all")
        assert entry is not None
        assert entry.K == 64
        assert entry.C == 16
        assert entry.atom_tensor.numel() == 64
        
        # Find by criteria
        llama_entries = lib2.find(family="llama")
        assert len(llama_entries) == 2
        
    finally:
        shutil.rmtree(tmpdir)
    
    print("  ✅ PASS")
    return True


def test_transfer_atoms():
    """Test atom transfer from source to target distribution."""
    print()
    print("=" * 60)
    print("TEST 3: Atom Transfer")
    print("=" * 60)
    
    # Source: "Llama 70B" distribution
    torch.manual_seed(100)
    source_weights = torch.randn(2048, dtype=torch.float32) * 0.15
    source_atoms = build_l0_atoms(source_weights, K=32, iters=3)
    source_coeffs = build_coeff_table(source_weights, source_atoms, C=8, iters=3)
    
    source_entry = build_entry_from_encoded(
        name="source-model", family="llama", variant="70b",
        component="all", atoms=source_atoms, coeffs=source_coeffs,
    )
    
    # Target: "Llama 8B" distribution (similar but different scale)
    torch.manual_seed(200)
    target_weights = torch.randn(2048, dtype=torch.float32) * 0.12  # slightly different std
    
    # Baseline: encode from scratch
    baseline_atoms = build_l0_atoms(target_weights, K=32, iters=3)
    baseline_coeffs = build_coeff_table(target_weights, baseline_atoms, C=8, iters=3)
    _, baseline_recon = wal_encode_v1(target_weights, baseline_atoms, baseline_coeffs)
    baseline_mse = (target_weights - baseline_recon).pow(2).mean().item()
    
    # Transfer: use source atoms with fine-tuning
    transfer_atoms, transfer_coeffs, transfer_recon = encode_with_pretrained_atoms(
        target_weights, source_entry
    )
    transfer_mse = (target_weights - transfer_recon).pow(2).mean().item()
    
    print(f"  Baseline MSE: {baseline_mse:.8f}")
    print(f"  Transfer MSE: {transfer_mse:.8f}")
    print(f"  Ratio (transfer/baseline): {transfer_mse / baseline_mse:.4f}")
    
    # Transfer should be competitive (within 2× of baseline)
    ratio = transfer_mse / (baseline_mse + 1e-12)
    assert ratio < 2.0, f"Transfer quality too poor: ratio={ratio:.4f}"
    
    print("  ✅ PASS")
    return True


def test_direct_transfer():
    """Test direct transfer (scale-only, no fine-tuning)."""
    print()
    print("=" * 60)
    print("TEST 4: Direct Transfer (Scale-Only)")
    print("=" * 60)
    
    # Source
    torch.manual_seed(300)
    source_weights = torch.randn(1024, dtype=torch.float32) * 0.2
    source_atoms = build_l0_atoms(source_weights, K=16, iters=3)
    source_coeffs = build_coeff_table(source_weights, source_atoms, C=4, iters=3)
    
    source_entry = build_entry_from_encoded(
        name="source", family="llama", variant="70b",
        component="mlp", atoms=source_atoms, coeffs=source_coeffs,
    )
    
    # Target with different scale
    torch.manual_seed(400)
    target_weights = torch.randn(1024, dtype=torch.float32) * 0.08
    
    # Direct transfer
    scaled_atoms, recon = transfer_atoms_direct(target_weights, source_entry)
    mse = (target_weights - recon).pow(2).mean().item()
    
    # Baseline
    baseline_atoms = build_l0_atoms(target_weights, K=16, iters=3)
    baseline_coeffs = build_coeff_table(target_weights, baseline_atoms, C=4, iters=3)
    _, baseline_recon = wal_encode_v1(target_weights, baseline_atoms, baseline_coeffs)
    baseline_mse = (target_weights - baseline_recon).pow(2).mean().item()
    
    print(f"  Direct transfer MSE: {mse:.8f}")
    print(f"  Baseline MSE: {baseline_mse:.8f}")
    print(f"  Ratio: {mse / (baseline_mse + 1e-12):.4f}")
    
    # Direct transfer is worse but still usable
    ratio = mse / (baseline_mse + 1e-12)
    assert ratio < 5.0, f"Direct transfer too poor: ratio={ratio:.4f}"
    
    print("  ✅ PASS")
    return True


def test_evaluate_transfer():
    """Test full transfer evaluation."""
    print()
    print("=" * 60)
    print("TEST 5: Transfer Evaluation")
    print("=" * 60)
    
    torch.manual_seed(500)
    source_weights = torch.randn(1024, dtype=torch.float32) * 0.1
    source_atoms = build_l0_atoms(source_weights, K=16, iters=3)
    source_coeffs = build_coeff_table(source_weights, source_atoms, C=4, iters=3)
    
    source_entry = build_entry_from_encoded(
        name="eval-source", family="llama", variant="70b",
        component="all", atoms=source_atoms, coeffs=source_coeffs,
    )
    
    torch.manual_seed(600)
    target_weights = torch.randn(1024, dtype=torch.float32) * 0.11
    
    baseline_atoms = build_l0_atoms(target_weights, K=16, iters=3)
    baseline_coeffs = build_coeff_table(target_weights, baseline_atoms, C=4, iters=3)
    
    metrics = evaluate_transfer(target_weights, source_entry, baseline_atoms, baseline_coeffs)
    
    print(f"  Baseline MSE: {metrics['baseline_mse']:.8f}")
    print(f"  Transfer MSE: {metrics['transfer_mse']:.8f}")
    print(f"  MSE ratio: {metrics['mse_ratio']:.4f}")
    print(f"  Transfer better: {metrics['transfer_better']}")
    print(f"  Max abs diff baseline: {metrics['max_abs_diff_baseline']:.6f}")
    print(f"  Max abs diff transfer: {metrics['max_abs_diff_transfer']:.6f}")
    
    assert 'baseline_mse' in metrics
    assert 'transfer_mse' in metrics
    assert 'mse_ratio' in metrics
    
    print("  ✅ PASS")
    return True


def test_library_query():
    """Test library querying."""
    print()
    print("=" * 60)
    print("TEST 6: Library Query")
    print("=" * 60)
    
    lib = create_default_library()
    
    entries = [
        ("llama-70b-attn", "llama", "70b", "attention"),
        ("llama-70b-mlp", "llama", "70b", "mlp"),
        ("llama-8b-attn", "llama", "8b", "attention"),
        ("llama-8b-mlp", "llama", "8b", "mlp"),
        ("mistral-7b-attn", "mistral", "7b", "attention"),
    ]
    
    for name, family, variant, component in entries:
        atoms = torch.randn(32, dtype=torch.float32)
        coeffs = torch.linspace(0.5, 2.0, 8, dtype=torch.float32)
        entry = build_entry_from_encoded(
            name=name, family=family, variant=variant,
            component=component, atoms=atoms, coeffs=coeffs,
        )
        lib.add_entry(entry)
    
    # Query by family
    llama = lib.find(family="llama")
    assert len(llama) == 4
    
    # Query by variant
    seven_b = lib.find(variant="70b")
    assert len(seven_b) == 2
    
    # Query by component
    attn = lib.find(component="attention")
    assert len(attn) == 3
    
    # Combined query
    llama_attn = lib.find(family="llama", component="attention")
    assert len(llama_attn) == 2
    
    print(f"  Total entries: {len(lib.entries)}")
    print(f"  Llama entries: {len(llama)}")
    print(f"  Attention entries: {len(attn)}")
    print(f"  Llama+Attention entries: {len(llama_attn)}")
    
    print("  ✅ PASS")
    return True


def main():
    print("\n" + "=" * 60)
    print("M79: WAL v1 Standard Library Prototype (Phase 8)")
    print("=" * 60 + "\n")
    
    results = []
    
    tests = [
        ("Build Entry", test_build_entry),
        ("Library Save/Load", test_library_save_load),
        ("Atom Transfer", test_transfer_atoms),
        ("Direct Transfer", test_direct_transfer),
        ("Transfer Evaluation", test_evaluate_transfer),
        ("Library Query", test_library_query),
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
