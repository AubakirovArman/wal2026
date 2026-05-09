"""WAL compression: codebook dedup, uint8 packing, row-scale quantization."""
import torch
from typing import Tuple, Optional
from .isa import ProgramBuffer


def compress_codebook(prog: ProgramBuffer) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Deduplicate programs into a codebook.
    
    Returns:
        codebook: [num_unique, lmax, 2] int64 — unique programs
        ids: [N] uint8 or uint16 — program IDs per weight
        num_unique: number of unique programs
    """
    N, lmax = prog.N, prog.lmax
    # Flatten programs into keys
    indices = prog.indices.long()  # [N, lmax]
    signs = (prog.signs + 1).long()  # [N, lmax], shift to {0,1,2}
    
    # Pack each program into a single int64 key
    max_idx = indices.max().item() + 1
    base = max(max_idx, 3)
    keys = torch.zeros(N, dtype=torch.int64)
    for s in range(lmax):
        keys = keys * (base * 3) + indices[:, s] * 3 + signs[:, s]
    
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    num_unique = unique_keys.shape[0]
    
    # Decode unique keys back to programs
    codebook = torch.zeros(num_unique, lmax, 2, dtype=torch.int64)
    temp = unique_keys.clone()
    for s in range(lmax - 1, -1, -1):
        codebook[:, s, 1] = (temp % 3) - 1  # sign
        temp = temp // 3
        codebook[:, s, 0] = temp % base  # idx
        temp = temp // base
    
    if num_unique <= 255:
        ids = inverse.to(torch.uint8)
    elif num_unique <= 65535:
        ids = inverse.to(torch.uint16)
    else:
        ids = inverse.to(torch.int32)
    
    return codebook, ids, num_unique


def decompress_codebook(codebook: torch.Tensor, ids: torch.Tensor, num_unique: int, lmax: int) -> ProgramBuffer:
    """Reconstruct ProgramBuffer from codebook + ids."""
    ids = ids.long()
    indices = codebook[ids, :, 0].to(torch.uint8)
    signs = codebook[ids, :, 1].to(torch.int8)
    return ProgramBuffer(indices, signs, lmax)


def compress_uint8_pack(prog: ProgramBuffer, K: int) -> torch.Tensor:
    """Pack each instruction into 1 byte.
    
    Encoding: byte = (sign + 1) * K + idx
    Requires K * 3 <= 255, i.e. K <= 85.
    
    Returns uint8 tensor [N, lmax].
    """
    assert K * 3 <= 255, f"K={K} too large for uint8 pack (max K=85)"
    signs_shifted = (prog.signs + 1).long()  # {0,1,2}
    codes = signs_shifted * K + prog.indices.long()
    return codes.to(torch.uint8)


def decompress_uint8_pack(codes: torch.Tensor, K: int, lmax: int) -> ProgramBuffer:
    """Unpack uint8 codes back to ProgramBuffer."""
    codes = codes.long()
    signs = (codes // K - 1).to(torch.int8)
    indices = (codes % K).to(torch.uint8)
    return ProgramBuffer(indices, signs, lmax)


def quantize_row_scales(row_scale: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float]:
    """Quantize row scales to lower precision.
    
    Args:
        row_scale: [M, 1] float32 positive scales
        bits: 8 or 16
    
    Returns:
        quantized: uint8 or uint16 tensor
        scale_factor: float to dequantize
    """
    max_val = row_scale.max().item()
    if bits == 8:
        q = (row_scale / max_val * 255).clamp(0, 255).to(torch.uint8)
        return q, max_val / 255.0
    elif bits == 16:
        q = (row_scale / max_val * 65535).clamp(0, 65535).to(torch.uint16)
        return q, max_val / 65535.0
    else:
        raise ValueError(f"bits must be 8 or 16, got {bits}")


def compute_compressed_size(
    prog: ProgramBuffer,
    K: int,
    row_scale_shape: Tuple[int, ...],
    use_codebook: bool = True,
    use_uint8_pack: bool = False,
    row_scale_bits: int = 16,
) -> dict:
    """Estimate compressed size in bytes."""
    N, lmax = prog.N, prog.lmax
    
    # Atom table
    atom_bytes = K * 4  # fp32
    
    # Programs
    if use_codebook:
        codebook, ids, num_unique = compress_codebook(prog)
        codebook_bytes = num_unique * lmax * 2 * 1  # uint8 per entry
        if num_unique <= 255:
            id_bytes = N * 1
        elif num_unique <= 65535:
            id_bytes = N * 2
        else:
            id_bytes = N * 4
        prog_bytes = codebook_bytes + id_bytes
        num_unique_val = num_unique
    elif use_uint8_pack and K * 3 <= 255:
        prog_bytes = N * lmax * 1  # uint8 per instruction
        num_unique_val = None
    else:
        prog_bytes = N * lmax * 2  # uint8 idx + int8 sign
        num_unique_val = None
    
    # Row scales
    num_rows = row_scale_shape[0]
    if row_scale_bits == 8:
        scale_bytes = num_rows * 1 + 4  # quantized + scale factor
    elif row_scale_bits == 16:
        scale_bytes = num_rows * 2 + 4
    else:
        scale_bytes = num_rows * 4
    
    total = atom_bytes + prog_bytes + scale_bytes
    original = N * 2  # bf16
    
    return {
        'total_bytes': total,
        'original_bytes': original,
        'compression_ratio': original / total,
        'atom_bytes': atom_bytes,
        'prog_bytes': prog_bytes,
        'scale_bytes': scale_bytes,
        'num_unique': num_unique_val,
    }
