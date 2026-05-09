"""WAL v2 Decoder: reconstruct weights from programs + atom/coeff tables."""
import torch
from .isa import ProgramBufferV2, AtomTable, CoeffTable


def wal_decode_v2(prog: ProgramBufferV2, atoms: AtomTable, coeffs: CoeffTable) -> torch.Tensor:
    """Decode WAL v2 programs to reconstructed weights.
    
    Args:
        prog: ProgramBufferV2 with encoded programs
        atoms: AtomTable
        coeffs: CoeffTable
        
    Returns:
        recon: [N] or [M, N] reconstructed weights
    """
    return prog.decode(atoms, coeffs).reshape(prog.shape)
