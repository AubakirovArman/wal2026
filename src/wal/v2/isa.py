"""WAL v2 ISA: Single-call program with continuous coefficients.

Program structure:
    weight = atom[atom_id] * coeff[coeff_id] + residual

This is simpler than WAL-0 (1 atom call vs 2) but more expressive
because coefficients are continuous (quantized to C levels) rather
than ternary {-1, 0, +1}.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AtomTable:
    """Shared atom table for a parameter."""
    values: torch.Tensor  # [K] float32
    
    @property
    def K(self) -> int:
        return self.values.shape[0]


@dataclass  
class CoeffTable:
    """Quantized coefficient levels.
    
    C levels stored as float32. Encoding uses uint4 (0..C-1).
    Typical: C=16 (4 bits), values learned via Lloyd-Max.
    """
    values: torch.Tensor  # [C] float32
    
    @property
    def C(self) -> int:
        return self.values.shape[0]


@dataclass
class WALProgram:
    """A single WAL v2 program."""
    atom_id: int      # uint8, index into AtomTable
    coeff_id: int     # uint4, index into CoeffTable  
    residual: float = 0.0  # optional float16 residual for exactness
    has_residual: bool = False


class ProgramBufferV2:
    """Stores WAL v2 programs with spatial layout.
    
    Shape: [N] for atom_ids, coeff_ids, residuals.
    N = number of weights.
    """
    def __init__(
        self,
        atom_ids: torch.Tensor,      # [N] uint8
        coeff_ids: torch.Tensor,     # [N] uint8 (only lower 4 bits used)
        residuals: torch.Tensor,     # [N] float32
        has_residual: torch.Tensor,  # [N] bool
        shape: Tuple[int, ...],      # original matrix shape
    ):
        assert atom_ids.dtype == torch.uint8
        assert coeff_ids.dtype == torch.uint8
        assert atom_ids.shape == coeff_ids.shape == residuals.shape == has_residual.shape
        self.atom_ids = atom_ids
        self.coeff_ids = coeff_ids
        self.residuals = residuals
        self.has_residual = has_residual
        self.shape = shape
        self.N = atom_ids.shape[0]
    
    @classmethod
    def empty(cls, N: int, shape: Tuple[int, ...], device='cpu'):
        return cls(
            atom_ids=torch.zeros(N, dtype=torch.uint8, device=device),
            coeff_ids=torch.zeros(N, dtype=torch.uint8, device=device),
            residuals=torch.zeros(N, dtype=torch.float32, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=shape,
        )
    
    def to(self, device):
        return ProgramBufferV2(
            atom_ids=self.atom_ids.to(device),
            coeff_ids=self.coeff_ids.to(device),
            residuals=self.residuals.to(device),
            has_residual=self.has_residual.to(device),
            shape=self.shape,
        )
    
    def cpu(self):
        return self.to('cpu')
    
    def get_program(self, row: int, col: int) -> WALProgram:
        idx = row * self.shape[1] + col
        return WALProgram(
            atom_id=int(self.atom_ids[idx].item()),
            coeff_id=int(self.coeff_ids[idx].item()),
            residual=float(self.residuals[idx].item()),
            has_residual=bool(self.has_residual[idx].item()),
        )
    
    def decode(self, atoms: AtomTable, coeffs: CoeffTable) -> torch.Tensor:
        """Decode all programs to reconstructed weights."""
        device = atoms.values.device
        atom_ids = self.atom_ids.to(device).long()
        coeff_ids = self.coeff_ids.to(device).long()
        
        recon = atoms.values[atom_ids] * coeffs.values[coeff_ids]
        if self.has_residual.any():
            recon += self.residuals.to(device) * self.has_residual.to(device)
        return recon
