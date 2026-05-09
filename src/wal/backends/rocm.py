#!/usr/bin/env python3
"""WAL ROCm Backend — AMD GPU decode via PyTorch ROCm/HIP.

Phase 9: ROCm backend for AMD GPUs.

Status: SCAFFOLD — implements the interface but is not tested on real
AMD hardware (no AMD GPU in current environment).

ROCm uses the same PyTorch CUDA API (torch.cuda.*) when built with
ROCm support, so this backend delegates to the same code as CUDA.
The difference is the underlying HIP runtime instead of CUDA runtime.
"""
import time
import torch
import numpy as np
from .base import WALBackend


class ROCmBackend(WALBackend):
    """ROCm/HIP backend for AMD GPUs.
    
    ROCm builds of PyTorch use torch.cuda.* APIs but compile to
    HIP kernels instead of CUDA kernels. This backend follows the
    same pattern as CUDA but checks for ROCm availability.
    
    To detect ROCm vs CUDA:
    - torch.version.hip is not None when using ROCm
    """
    
    @property
    def name(self) -> str:
        return "rocm"
    
    @property
    def is_available(self) -> bool:
        # ROCm uses torch.cuda but with hip runtime
        return torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    def decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape, device=None):
        """Decode using ROCm/HIP backend.
        
        Uses the same PyTorch GPU ops as CUDA, but compiles to HIP.
        """
        from wal.v1.decoder import wal_decode_v1
        
        device = device or torch.device("cuda:0")  # ROCm uses cuda device naming
        
        atom_ids = atom_ids.to(device)
        coeff_ids = coeff_ids.to(device)
        
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
        """Benchmark decode on ROCm."""
        if not self.is_available:
            return float('inf')
        
        device = device or torch.device("cuda:0")
        
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
