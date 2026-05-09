"""WAL-0 Decoder: reconstruct weights from programs + atom table."""
import torch
from .isa import ProgramBuffer


def wal_decode_scalar_torch(prog: ProgramBuffer, atoms: torch.Tensor) -> torch.Tensor:
    """Decode WAL-0 programs using PyTorch (CPU/GPU).
    
    Args:
        prog: ProgramBuffer with shape [N, lmax]
        atoms: [K] atom table
    
    Returns:
        recon: [N] reconstructed weights
    """
    device = atoms.device
    indices = prog.indices.to(device).long()
    signs = prog.signs.to(device).float()
    
    # Gather atoms and apply signs: [N, lmax]
    gathered = atoms[indices] * signs
    recon = gathered.sum(dim=1)  # [N]
    return recon
