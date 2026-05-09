#!/usr/bin/env python3
"""WAL CUDA Backend — GPU-accelerated decode via PyTorch CUDA.

Phase 9: CUDA backend using PyTorch GPU ops (default production path).
"""
import time
import torch
import numpy as np
from .base import WALBackend


class CUDABackend(WALBackend):
    """CUDA backend using PyTorch GPU operations.
    
    This is the default production backend. It uses:
    - PyTorch tensor indexing on GPU
    - Optional Triton kernels for fused decode
    """
    
    @property
    def name(self) -> str:
        return "cuda"
    
    @property
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape, device=None):
        """Decode using PyTorch GPU ops.
        
        Args:
            atom_ids: [N] uint8
            coeff_ids: [N] uint8
            atom_table: AtomTableV1
            coeff_table: CoeffTable
            shape: Target shape
            device: CUDA device (default: cuda:0)
        
        Returns:
            torch.Tensor on CUDA
        """
        from wal.v1.decoder import precompute_flat_atoms, wal_decode_v1
        
        device = device or torch.device("cuda:0")
        
        # Move inputs to GPU
        atom_ids = atom_ids.to(device)
        coeff_ids = coeff_ids.to(device)
        
        # Use existing v1 decoder (already GPU-compatible)
        decoded = wal_decode_v1(
            prog=type('obj', (object,), {
                'atom_ids': atom_ids,
                'coeff_ids': coeff_ids,
                'residuals': torch.empty(0, dtype=torch.float16, device=device),
                'has_residual': torch.zeros(atom_ids.numel(), dtype=torch.bool, device=device),
                'shape': shape,
                'N': atom_ids.numel(),
            })(),
            atom_table=atom_table,
            coeff_values=(coeff_table if torch.is_tensor(coeff_table) else coeff_table.values).to(device),
        )
        
        return decoded
    
    def benchmark_decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape,
                         num_runs: int = 10, device=None) -> float:
        """Benchmark decode speed on CUDA.
        
        Returns:
            Median decode time in milliseconds
        """
        if not self.is_available:
            return float('inf')
        
        device = device or torch.device("cuda:0")
        
        # Move to GPU and warmup
        _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape, device)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape, device)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return float(np.median(times))
