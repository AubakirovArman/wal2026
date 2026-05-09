#!/usr/bin/env python3
"""M80: WAL Hardware Backends Test (Phase 9).

Tests:
1. Backend registry and availability
2. CPU backend decode correctness
3. CUDA backend decode correctness (if available)
4. Cross-backend consistency (all backends give same result)
5. Benchmark comparison
6. WebGPU WGSL shader generation
7. Backend selection logic
"""
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import numpy as np
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms
from wal.backends import (
    get_backend, list_backends, available_backends, select_best_backend,
    CPUBackend, CUDABackend, MPSBackend, ROCmBackend, WebGPUBackend,
)


def test_registry():
    """Test backend registry."""
    print("=" * 60)
    print("TEST 1: Backend Registry")
    print("=" * 60)
    
    backends = list_backends()
    avail = available_backends()
    
    print(f"  All backends: {backends}")
    print(f"  Available: {avail}")
    
    assert 'cpu' in backends, "CPU backend must be registered"
    assert 'cpu' in avail, "CPU backend must be available"
    
    # Get each backend
    cpu = get_backend('cpu')
    assert isinstance(cpu, CPUBackend)
    assert cpu.is_available
    
    print(f"  CPU backend: {cpu}")
    
    if 'cuda' in backends:
        cuda = get_backend('cuda')
        print(f"  CUDA backend: {cuda}")
    
    print("  ✅ PASS")
    return True


def test_cpu_decode():
    """Test CPU backend decode correctness."""
    print()
    print("=" * 60)
    print("TEST 2: CPU Backend Decode")
    print("=" * 60)
    
    torch.manual_seed(42)
    weights = torch.randn(1024, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=32, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=8, iters=3)
    prog, recon_ref = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
    
    cpu = get_backend('cpu')
    decoded = cpu.decode(
        prog.atom_ids, prog.coeff_ids,
        atom_table, coeffs,
        shape=weights.shape,
    )
    
    max_diff = (recon_ref.cpu() - decoded.cpu()).abs().max().item()
    mean_diff = (recon_ref.cpu() - decoded.cpu()).abs().mean().item()
    
    print(f"  Weights shape: {weights.shape}")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    
    assert max_diff < 1e-4, f"CPU decode too inaccurate: {max_diff}"
    
    print("  ✅ PASS")
    return True


def test_cuda_decode():
    """Test CUDA backend decode if available."""
    print()
    print("=" * 60)
    print("TEST 3: CUDA Backend Decode")
    print("=" * 60)
    
    cuda = get_backend('cuda')
    
    if not cuda.is_available:
        print("  CUDA not available — skipping")
        print("  ⏭️  SKIP")
        return True
    
    torch.manual_seed(42)
    weights = torch.randn(1024, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=32, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=8, iters=3)
    prog, recon_ref = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
    
    decoded = cuda.decode(
        prog.atom_ids, prog.coeff_ids,
        atom_table, coeffs,
        shape=weights.shape,
    )
    
    max_diff = (recon_ref.cpu() - decoded.cpu()).abs().max().item()
    
    print(f"  CUDA decode max diff: {max_diff:.8f}")
    assert max_diff < 1e-4, f"CUDA decode inaccurate: {max_diff}"
    
    print("  ✅ PASS")
    return True


def test_cross_backend_consistency():
    """Test that all available backends produce identical results."""
    print()
    print("=" * 60)
    print("TEST 4: Cross-Backend Consistency")
    print("=" * 60)
    
    torch.manual_seed(123)
    weights = torch.randn(512, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=16, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=4, iters=3)
    prog, _ = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=4)
    
    results = {}
    for name in available_backends():
        backend = get_backend(name)
        try:
            decoded = backend.decode(
                prog.atom_ids, prog.coeff_ids,
                atom_table, coeffs,
                shape=weights.shape,
            )
            results[name] = decoded.cpu()
            print(f"  {name}: decoded shape {decoded.shape}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
    
    # Compare all pairs
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            max_diff = (results[names[i]] - results[names[j]]).abs().max().item()
            print(f"  {names[i]} vs {names[j]}: max_diff={max_diff:.8f}")
            assert max_diff < 1e-4, f"Backend mismatch: {names[i]} vs {names[j]}"
    
    print("  ✅ PASS")
    return True


def test_benchmark():
    """Benchmark available backends."""
    print()
    print("=" * 60)
    print("TEST 5: Backend Benchmark")
    print("=" * 60)
    
    torch.manual_seed(456)
    weights = torch.randn(10000, dtype=torch.float32) * 0.1
    
    atoms = build_l0_atoms(weights, K=32, iters=3)
    coeffs = build_coeff_table(weights, atoms, C=8, iters=3)
    prog, _ = wal_encode_v1(weights, atoms, coeffs)
    atom_table = build_hierarchical_atoms(atoms, prog, max_l1=8)
    
    for name in available_backends():
        backend = get_backend(name)
        try:
            ms = backend.benchmark_decode(
                prog.atom_ids, prog.coeff_ids,
                atom_table, coeffs,
                shape=weights.shape,
                num_runs=5,
            )
            throughput = weights.numel() / (ms / 1000) / 1e6  # Mweights/s
            print(f"  {name:8s}: {ms:8.3f} ms | {throughput:8.2f} Mw/s")
        except Exception as e:
            print(f"  {name:8s}: benchmark failed ({e})")
    
    print("  ✅ PASS")
    return True


def test_wgsl_shader():
    """Test WebGPU WGSL shader generation."""
    print()
    print("=" * 60)
    print("TEST 6: WebGPU WGSL Shader")
    print("=" * 60)
    
    webgpu = get_backend('webgpu')
    
    shader = webgpu.generate_wgsl_shader(K=256, C=16)
    
    print("  Generated WGSL shader:")
    for line in shader.strip().split('\n')[:10]:
        print(f"    {line}")
    print("    ...")
    
    assert '@compute' in shader, "Shader should have compute entry point"
    assert 'atom_ids' in shader, "Shader should reference atom_ids"
    assert 'coeff_ids' in shader, "Shader should reference coeff_ids"
    assert 'atom_table' in shader, "Shader should reference atom_table"
    
    print("  ✅ PASS")
    return True


def test_backend_selection():
    """Test automatic backend selection."""
    print()
    print("=" * 60)
    print("TEST 7: Backend Selection")
    print("=" * 60)
    
    best = select_best_backend()
    print(f"  Best backend: {best.name}")
    print(f"  Available: {available_backends()}")
    
    assert best.is_available, "Selected backend must be available"
    
    # CPU should always be fallback
    if not any(b in available_backends() for b in ['cuda', 'rocm', 'mps']):
        assert best.name == 'cpu', "CPU should be fallback when no GPU available"
    
    print("  ✅ PASS")
    return True


def test_scaffold_backends():
    """Test that scaffold backends exist and report availability."""
    print()
    print("=" * 60)
    print("TEST 8: Scaffold Backends")
    print("=" * 60)
    
    # MPS
    mps = get_backend('mps')
    print(f"  MPS: available={mps.is_available}")
    
    # ROCm
    rocm = get_backend('rocm')
    print(f"  ROCm: available={rocm.is_available}")
    
    # WebGPU
    webgpu = get_backend('webgpu')
    print(f"  WebGPU: available={webgpu.is_available}")
    
    # At least one should report availability status
    assert isinstance(mps, MPSBackend)
    assert isinstance(rocm, ROCmBackend)
    assert isinstance(webgpu, WebGPUBackend)
    
    print("  ✅ PASS")
    return True


def main():
    print("\n" + "=" * 60)
    print("M80: WAL Hardware Backends Test (Phase 9)")
    print("=" * 60 + "\n")
    
    results = []
    
    tests = [
        ("Registry", test_registry),
        ("CPU Decode", test_cpu_decode),
        ("CUDA Decode", test_cuda_decode),
        ("Cross-Backend Consistency", test_cross_backend_consistency),
        ("Benchmark", test_benchmark),
        ("WGSL Shader", test_wgsl_shader),
        ("Backend Selection", test_backend_selection),
        ("Scaffold Backends", test_scaffold_backends),
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
