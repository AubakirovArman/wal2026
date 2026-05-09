#!/usr/bin/env python3
"""M53b: Fused Triton encode kernel benchmark vs PyTorch loop encode."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal import build_atoms_kmeans, wal_encode_scalar
from wal.triton_encode import wal_encode_scalar_fused


def benchmark(N, K, lmax, device, repeats=20):
    torch.manual_seed(42)
    weights = torch.randn(N, device=device)
    samples = weights[torch.randperm(N)[:min(N, 1_000_000)]]
    atoms = build_atoms_kmeans(samples, K, iters=3, device=device).to(device)
    
    # Warmup Triton
    for _ in range(3):
        _ = wal_encode_scalar_fused(weights, atoms, lmax)
    torch.cuda.synchronize(device)
    
    # PyTorch encode
    t0 = time.time()
    for _ in range(repeats):
        _, recon_torch = wal_encode_scalar(weights, atoms, lmax)
    torch.cuda.synchronize(device)
    t_torch = (time.time() - t0) / repeats
    
    # Fused Triton encode
    t0 = time.time()
    for _ in range(repeats):
        recon_triton = wal_encode_scalar_fused(weights, atoms, lmax)
    torch.cuda.synchronize(device)
    t_triton = (time.time() - t0) / repeats
    
    # Quality
    max_err = (recon_torch - recon_triton).abs().max().item()
    rel_mse = ((recon_torch - recon_triton) ** 2).mean() / (recon_torch ** 2).mean()
    
    return {
        'torch_ms': t_torch * 1000,
        'triton_ms': t_triton * 1000,
        'torch_Mw/s': N / t_torch / 1e6,
        'triton_Mw/s': N / t_triton / 1e6,
        'speedup': t_torch / t_triton,
        'max_err': max_err,
        'rel_mse': rel_mse.item(),
    }


def main():
    device = torch.device('cuda:2')
    K, lmax = 128, 2
    
    print("=" * 60)
    print("M53b: Fused Triton Encode Kernel")
    print("=" * 60)
    
    for N in [1_000_000, 10_000_000, 50_000_000, 100_000_000]:
        print(f"\nN={N}, K={K}, lmax={lmax}")
        bench = benchmark(N, K, lmax, device, repeats=20)
        print(f"  PyTorch:  {bench['torch_ms']:.2f} ms  ({bench['torch_Mw/s']:.1f} Mw/s)")
        print(f"  Triton:   {bench['triton_ms']:.2f} ms  ({bench['triton_Mw/s']:.1f} Mw/s)")
        print(f"  Speedup:  {bench['speedup']:.1f}x")
        print(f"  Max err:  {bench['max_err']:.8f}")
        print(f"  relMSE:   {bench['rel_mse']:.2e}")
    
    print("\n" + "=" * 60)
    print("M53b: DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
