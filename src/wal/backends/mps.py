#!/usr/bin/env python3
"""WAL Metal/MPS Backend — Apple Silicon GPU decode.

Phase 9: Metal Performance Shaders (MPS) backend for Apple Silicon.

Status: SCAFFOLD — implements the interface but is not tested on real
Metal hardware (no Apple Silicon in current environment).
"""
import time
import torch
import numpy as np
from .base import WALBackend


class MPSBackend(WALBackend):
    """Metal Performance Shaders backend for Apple Silicon Macs.
    
    This backend uses PyTorch's MPS backend, which translates
    PyTorch ops into Metal Performance Shaders on macOS.
    
    Implementation strategy:
    1. Move inputs to MPS device
    2. Use PyTorch tensor indexing (translated to MPS)
    3. Element-wise ops (translated to Metal compute shaders)
    
    Note: MPS does not support uint8 indexing natively, so we
    convert to int64 before indexing.
    """
    
    @property
    def name(self) -> str:
        return "mps"
    
    @property
    def is_available(self) -> bool:
        return torch.backends.mps.is_available()
    
    def decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape, device=None):
        """Decode using MPS backend.
        
        Args:
            atom_ids: [N] uint8
            coeff_ids: [N] uint8
            atom_table: AtomTableV1
            coeff_table: CoeffTable
            shape: Target shape
            device: MPS device (default: mps)
        
        Returns:
            torch.Tensor on MPS
        """
        from wal.v1.decoder import wal_decode_v1
        
        device = device or torch.device("mps")
        
        # MPS does not support uint8 indexing well
        atom_ids = atom_ids.to(device).long()
        coeff_ids = coeff_ids.to(device).long()
        
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
        """Benchmark decode on MPS."""
        if not self.is_available:
            return float('inf')
        
        device = device or torch.device("mps")
        
        _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape, device)
        torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape, device)
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return float(np.median(times))
