"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M47: WAL-0 Runtime test — encode, decode, serialize, benchmark."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal import (
    wal_encode_scalar,
    build_atoms_kmeans,
    wal_decode_scalar_torch,
    pack_programs,
    unpack_programs,
    serialize_wal_state,
    deserialize_wal_state,
    WALModelState,
    WALParameterMeta,
)
from wal.triton_kernels import wal_decode_scalar_triton


def benchmark_decode(N, K, lmax, device, repeats=100):
    """Benchmark torch vs triton decode throughput."""
    indices = torch.randint(0, K, (N, lmax), dtype=torch.uint8, device=device)
    signs = torch.randint(-1, 2, (N, lmax), dtype=torch.int8, device=device)
    atoms = torch.randn(K, dtype=torch.float32, device=device)
    
    # Warmup Triton
    for _ in range(10):
        _ = wal_decode_scalar_triton(indices, signs, atoms)
    torch.cuda.synchronize(device)
    
    # Move to device for triton
    indices = indices.to(device)
    signs = signs.to(device)
    atoms = atoms.to(device)
    
    # Torch decode
    from wal.isa import ProgramBuffer
    prog = ProgramBuffer(indices, signs, lmax)
    t0 = time.time()
    for _ in range(repeats):
        out_torch = wal_decode_scalar_torch(prog, atoms)
    torch.cuda.synchronize(device)
    t_torch = (time.time() - t0) / repeats
    
    # Triton decode
    t0 = time.time()
    for _ in range(repeats):
        out_triton = wal_decode_scalar_triton(indices, signs, atoms)
    torch.cuda.synchronize(device)
    t_triton = (time.time() - t0) / repeats
    
    # Verify correctness
    max_err = (out_torch - out_triton).abs().max().item()
    
    throughput_torch = N / t_torch / 1e6  # Mweights/sec
    throughput_triton = N / t_triton / 1e6
    
    return {
        'torch_ms': t_torch * 1000,
        'triton_ms': t_triton * 1000,
        'torch_Mw/s': throughput_torch,
        'triton_Mw/s': throughput_triton,
        'max_err': max_err,
    }


def test_end_to_end():
    """Test encode → pack → unpack → decode → serialize → deserialize."""
    print("=" * 60)
    print("M47: WAL-0 Runtime End-to-End Test")
    print("=" * 60)
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    N = 1_000_000
    K = 128
    lmax = 2
    
    # Generate synthetic weights
    torch.manual_seed(42)
    weights = torch.randn(N, device=device)
    
    print(f"\n[1] Encoding {N} weights, K={K}, lmax={lmax}...")
    t0 = time.time()
    atoms = build_atoms_kmeans(weights, K, iters=5, device=device)
    atoms = atoms.to(device)
    prog, recon = wal_encode_scalar(weights, atoms, lmax)
    encode_time = time.time() - t0
    print(f"    Encode done in {encode_time:.2f}s")
    
    # Quality metrics
    rel_mse = ((weights - recon) ** 2).mean() / (weights ** 2).mean()
    print(f"    relMSE: {rel_mse.item():.6f}")
    
    # Pack/unpack test
    print(f"\n[2] Pack/unpack test...")
    codes = pack_programs(prog)
    print(f"    Codes dtype: {codes.dtype}, shape: {codes.shape}")
    prog2 = unpack_programs(codes, K, lmax)
    assert (prog.indices == prog2.indices).all()
    assert (prog.signs == prog2.signs).all()
    print("    Pack/unpack OK")
    
    # Decode test: torch vs triton
    print(f"\n[3] Decode test: torch vs triton...")
    recon_torch = wal_decode_scalar_torch(prog, atoms.to(device))
    recon_triton = wal_decode_scalar_triton(prog.indices.to(device), prog.signs.to(device), atoms.to(device))
    max_err = (recon_torch - recon_triton).abs().max().item()
    print(f"    Max error torch vs triton: {max_err:.8f}")
    assert max_err < 1e-5, f"Triton decode mismatch: {max_err}"
    
    # Serialize/deserialize test
    print(f"\n[4] Serialize/deserialize test...")
    meta = WALParameterMeta(
        name="test.weight",
        shape=[N],
        device=str(device),
        row_scale_shape=[1],
        offset=0,
        numel=N,
        is_encoded=True,
    )
    state = WALModelState(
        K=K, lmax=lmax, dtype_str='float32',
        atom_table=atoms.cpu(),
        programs=prog.cpu(),
        params=[meta],
    )
    blob = serialize_wal_state(state)
    state2 = deserialize_wal_state(blob)
    print(f"    Blob size: {len(blob)} bytes")
    print(f"    Original weights: {weights.numel() * 4} bytes (fp32)")
    print(f"    Compression: {weights.numel() * 4 / len(blob):.2f}x")
    
    recon3 = wal_decode_scalar_torch(state2.programs, state2.atom_table)
    max_err2 = (recon.cpu() - recon3).abs().max().item()
    print(f"    Max error after round-trip: {max_err2:.8f}")
    
    # Benchmark
    print(f"\n[5] Benchmark decode throughput (N={N}, K={K}, lmax={lmax})...")
    bench = benchmark_decode(N, K, lmax, device, repeats=100)
    print(f"    Torch:   {bench['torch_ms']:.3f} ms  ({bench['torch_Mw/s']:.1f} Mweights/s)")
    print(f"    Triton:  {bench['triton_ms']:.3f} ms  ({bench['triton_Mw/s']:.1f} Mweights/s)")
    print(f"    Speedup: {bench['torch_ms'] / bench['triton_ms']:.2f}x")
    print(f"    Max error: {bench['max_err']:.8f}")
    
    # Larger benchmark
    print(f"\n[6] Large benchmark (N=100M, K={K}, lmax={lmax})...")
    bench_large = benchmark_decode(100_000_000, K, lmax, device, repeats=20)
    print(f"    Torch:   {bench_large['torch_ms']:.3f} ms  ({bench_large['torch_Mw/s']:.1f} Mweights/s)")
    print(f"    Triton:  {bench_large['triton_ms']:.3f} ms  ({bench_large['triton_Mw/s']:.1f} Mweights/s)")
    print(f"    Speedup: {bench_large['torch_ms'] / bench_large['triton_ms']:.2f}x")
    
    print("\n" + "=" * 60)
    print("M47: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end()
