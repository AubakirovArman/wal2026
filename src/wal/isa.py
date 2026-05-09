"""WAL-0 ISA: Instruction Set Architecture for scalar weight programs."""
import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class WAL_ISA:
    """WAL-0 Instruction Set.
    
    Programs are sequences of atom calls with ternary coefficients.
    Each instruction is conceptually: ACC += atom[idx] * coeff
    We store programs as parallel arrays for SIMD efficiency.
    """
    K: int  # number of atoms in table
    lmax: int  # max program length
    coeff_values: Tuple[int, ...] = (-1, 0, +1)

    def coeff_to_int(self, c: float) -> int:
        """Map float coefficient to integer index."""
        if abs(c) < 0.5:
            return 1  # zero
        return 2 if c > 0 else 0  # pos, neg

    def int_to_coeff(self, i: int) -> float:
        """Map integer index back to coefficient."""
        return self.coeff_values[i]


class ProgramBuffer:
    """Stores WAL-0 programs as parallel arrays.
    
    Shape: [N, lmax] for indices and signs.
    N = number of weights.
    """
    def __init__(self, indices: torch.Tensor, signs: torch.Tensor, lmax: int):
        assert indices.dtype == torch.uint8 or indices.dtype == torch.int32
        assert signs.dtype == torch.int8
        assert indices.shape == signs.shape
        self.indices = indices  # [N, lmax] atom indices
        self.signs = signs      # [N, lmax] {-1, 0, +1}
        self.lmax = lmax
        self.N = indices.shape[0]

    @classmethod
    def empty(cls, N: int, lmax: int, device='cpu'):
        return cls(
            indices=torch.zeros(N, lmax, dtype=torch.uint8, device=device),
            signs=torch.zeros(N, lmax, dtype=torch.int8, device=device),
            lmax=lmax,
        )

    def to(self, device):
        return ProgramBuffer(self.indices.to(device), self.signs.to(device), self.lmax)

    def cpu(self):
        return self.to('cpu')


def pack_programs(prog: ProgramBuffer) -> torch.Tensor:
    """Pack (idx, sign) pairs into compact int16 codes.
    
    Encoding: code = (sign + 1) * K + idx  # sign in {-1,0,+1} -> {0,1,2}
    Returns int16 tensor [N, lmax] if K <= 256, else int32.
    """
    N, lmax = prog.N, prog.lmax
    K = prog.indices.max().item() + 1 if prog.indices.numel() > 0 else 1
    
    # sign: {-1, 0, +1} -> {0, 1, 2}
    sign_codes = (prog.signs + 1).long()  # [N, lmax]
    codes = sign_codes * K + prog.indices.long()  # [N, lmax]
    
    max_code = K * 3
    if max_code <= 255:
        return codes.to(torch.uint8)
    elif max_code <= 32767:
        return codes.to(torch.int16)
    else:
        return codes.to(torch.int32)


def unpack_programs(codes: torch.Tensor, K: int, lmax: int) -> ProgramBuffer:
    """Unpack compact codes back to (idx, sign) arrays."""
    codes = codes.long()
    sign_codes = codes // K  # {0, 1, 2}
    indices = (codes % K).to(torch.uint8)
    signs = (sign_codes - 1).to(torch.int8)  # back to {-1, 0, +1}
    return ProgramBuffer(indices, signs, lmax)
