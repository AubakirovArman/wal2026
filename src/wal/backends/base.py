#!/usr/bin/env python3
"""WAL Backend Base — abstract interface for hardware backends.

Phase 9: Hardware Backends (Metal, ROCm, WebGPU, CPU SIMD).
"""
from abc import ABC, abstractmethod
from typing import Tuple
import torch


class WALBackend(ABC):
    """Abstract base class for WAL hardware backends.
    
    Each backend provides encode and decode operations for WAL programs.
    Backends are responsible for device placement and optimization.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'cpu', 'cuda', 'mps', 'rocm')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is available on the current system."""
        pass
    
    @abstractmethod
    def decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape, device=None):
        """Decode WAL programs to dense tensor.
        
        Args:
            atom_ids: [N] uint8
            coeff_ids: [N] uint8
            atom_table: AtomTableV1
            coeff_table: CoeffTable
            shape: Target shape
            device: Target device (backend-specific)
        
        Returns:
            Decoded tensor
        """
        pass
    
    @abstractmethod
    def benchmark_decode(self, atom_ids, coeff_ids, atom_table, coeff_table, shape,
                         num_runs: int = 10, device=None) -> float:
        """Benchmark decode speed.
        
        Returns:
            Median decode time in milliseconds
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, available={self.is_available})"
