#!/usr/bin/env python3
"""WAL Framework — Phase 9: Backend selection and benchmark."""


def benchmark_backend(backend: str = "auto", device: str = "cuda"):
    """Benchmark WAL decode on selected backend.
    
    Args:
        backend: Backend name ('auto', 'cpu', 'cuda', etc.)
        device: Device for benchmarking
    """
    from wal.backends import select_best_backend, available_backends
    import torch
    import time
    
    print("[Backend] Available backends:")
    for name in available_backends():
        print(f"  - {name}")
    
    if backend == "auto":
        be = select_best_backend()
        print(f"[Backend] Auto-selected: {be.__class__.__name__}")
    else:
        from wal.backends import get_backend
        be = get_backend(backend)
        print(f"[Backend] Selected: {backend}")
    
    # Benchmark
    print("[Backend] Benchmarking...")
    shape = (4096, 4096)
    N = shape[0] * shape[1]
    K, C = 256, 16
    
    atom_ids = torch.randint(0, K, (N,), dtype=torch.uint8, device=device)
    coeff_ids = torch.randint(0, C, (N,), dtype=torch.uint8, device=device)
    
    import numpy as np
    atoms = np.random.randn(K).astype(np.float32)
    coeffs = np.random.randn(C).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        _ = be.decode(atom_ids, coeff_ids, atoms, coeffs, shape)
    
    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    iterations = 10
    for _ in range(iterations):
        _ = be.decode(atom_ids, coeff_ids, atoms, coeffs, shape)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start
    
    total_weights = N * iterations
    speed = total_weights / elapsed / 1e6
    print(f"[Backend] Decoded {total_weights} weights in {elapsed:.3f}s")
    print(f"[Backend] Speed: {speed:.1f} Mw/s")
