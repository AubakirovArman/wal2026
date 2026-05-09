#!/usr/bin/env python3
"""WAL CPU Backend — SIMD-accelerated decode via NumPy/PyTorch CPU.

Phase 9: CPU backend using NumPy vectorized operations.
Works on any system with PyTorch/NumPy installed.
"""
import time
import torch
import numpy as np
from .base import WALBackend


class CPUBackend(WALBackend):
    """CPU backend using NumPy vectorized ops for decode.
    
    This backend:
    1. Converts WAL programs to NumPy arrays
    2. Uses vectorized lookup for atom_id → atom_value
    3. Uses vectorized lookup for coeff_id → coeff_value
    4. Element-wise multiplication + residual addition
    
    The NumPy path can be faster than naive Python loops for large tensors.
    """
    
    @property
    def name(self) -> str:
        return "cpu"
    
    @property
    def is_available(self) -> bool:
        return True  # CPU is always available
    
    def decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape, device=None):
        """Decode using NumPy vectorized ops.
        
        Args:
            atom_ids: [N] uint8 (torch tensor)
            coeff_ids: [N] uint8 (torch tensor)
            atom_table: AtomTableV1
            coeff_table: CoeffTable
            shape: Target shape
            device: Ignored for CPU backend
        
        Returns:
            torch.Tensor on CPU
        """
        # Precompute flat atoms
        from wal.v1.decoder import precompute_flat_atoms
        flat_atoms = precompute_flat_atoms(atom_table).cpu().numpy()
        
        # Convert to numpy
        atom_ids_np = atom_ids.cpu().numpy().astype(np.int64)
        coeff_ids_np = coeff_ids.cpu().numpy().astype(np.int64)
        coeff_values = coeff_table.cpu().numpy() if torch.is_tensor(coeff_table) else coeff_table.values.cpu().numpy()
        
        # Vectorized decode
        atom_vals = flat_atoms[atom_ids_np]
        coeff_vals = coeff_values[coeff_ids_np]
        decoded = atom_vals * coeff_vals
        
        # Add residuals if present
        # Note: residuals are passed separately in a real implementation
        # Here we assume no residuals for simplicity
        
        # Reshape
        decoded = decoded.reshape(shape)
        
        return torch.from_numpy(decoded).float()
    
    def benchmark_decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape,
                         num_runs: int = 10, device=None) -> float:
        """Benchmark decode speed on CPU.
        
        Returns:
            Median decode time in milliseconds
        """
        times = []
        for _ in range(num_runs):
            # Warmup
            _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape)
            
            start = time.perf_counter()
            _ = self.decode(atom_ids, coeff_ids, atom_table, coeff_table, shape)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return float(np.median(times))
