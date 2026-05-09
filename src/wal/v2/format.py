"""WAL v2 Binary Format: compact serialization for model shipping.

Format v0.1:
  Header (32 bytes):
    magic:       b'WAL2' (4 bytes)
    version:     uint16 (1)
    K:           uint16
    C:           uint16
    flags:       uint16 (bit 0: has_residuals, bit 1: packed_coeffs)
    N_weights:   uint64
    M_rows:      uint32
    reserved:    8 bytes

  Atom Table:    K × 4 bytes (float32)
  Coeff Table:   C × 4 bytes (float32)

  Programs:
    atom_ids:       N × 1 byte (uint8)
    coeff_ids:      ceil(N/2) bytes (uint4, packed 2 per byte) if packed
                    OR N × 1 byte (uint8) if unpacked
    has_residual:   ceil(N/8) bytes (bitmap)
    residual_count: uint32
    residual_indices: residual_count × uint32 (if count > 0)
    residual_values:  residual_count × 2 bytes (float16, if count > 0)

  Row Scales:    M × 4 bytes (float32)
  Metadata:      JSON length-prefixed
"""
import json
import struct
import torch
import numpy as np
from typing import Dict, Tuple
from .isa import ProgramBufferV2, AtomTable, CoeffTable


MAGIC = b'WAL2'
VERSION = 1

# Flags
FLAG_HAS_RESIDUALS = 1 << 0
FLAG_PACKED_COEFFS = 1 << 1


def _pack_uint4(values: torch.Tensor) -> torch.Tensor:
    """Pack uint8 values (0..15) into uint4 pairs."""
    N = values.numel()
    padded = values
    if N % 2 == 1:
        padded = torch.cat([values, torch.zeros(1, dtype=torch.uint8)])
    even = padded[0::2] & 0x0F
    odd = padded[1::2] & 0x0F
    return (even << 4) | odd


def _unpack_uint4(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint4 pairs into uint8 values."""
    even = (packed >> 4) & 0x0F
    odd = packed & 0x0F
    values = torch.stack([even, odd], dim=1).flatten()[:N]
    return values


def _pack_bitmap(bool_tensor: torch.Tensor) -> bytes:
    """Pack boolean tensor into bit-packed bytes."""
    arr = bool_tensor.cpu().numpy().astype(np.uint8)
    return np.packbits(arr).tobytes()


def _unpack_bitmap(data: bytes, N: int) -> torch.Tensor:
    """Unpack bit-packed bytes into boolean tensor."""
    arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:N]
    return torch.from_numpy(arr).bool()


def serialize_wal_v2(
    prog: ProgramBufferV2,
    atoms: AtomTable,
    coeffs: CoeffTable,
    row_scales: torch.Tensor,
    metadata: Dict = None,
) -> bytes:
    """Serialize WAL v2 state to compact binary."""
    K = atoms.K
    C = coeffs.C
    N = prog.N
    M = row_scales.numel()

    has_residuals = prog.has_residual.any().item()
    flags = FLAG_PACKED_COEFFS
    if has_residuals:
        flags |= FLAG_HAS_RESIDUALS

    # Header (32 bytes)
    header = struct.pack(
        '<4sHHHHQIB7s',
        MAGIC, VERSION, K, C, flags, N, M, 0, b'\x00' * 7,
    )
    assert len(header) == 32

    # Tables
    atom_bytes = atoms.values.cpu().float().numpy().tobytes()
    coeff_bytes = coeffs.values.cpu().float().numpy().tobytes()

    # Programs
    atom_id_bytes = prog.atom_ids.cpu().numpy().tobytes()

    coeff_packed = _pack_uint4(prog.coeff_ids.cpu())
    coeff_id_bytes = coeff_packed.numpy().tobytes()

    # Residuals
    residual_bitmap = _pack_bitmap(prog.has_residual.cpu())
    residual_count = int(prog.has_residual.sum().item())

    if residual_count > 0:
        residual_indices = prog.has_residual.nonzero(as_tuple=False).flatten().cpu()
        residual_values = prog.residuals[prog.has_residual].cpu().half()

        residual_section = (
            struct.pack('<I', residual_count)
            + residual_bitmap
            + residual_indices.numpy().astype(np.uint32).tobytes()
            + residual_values.numpy().tobytes()
        )
    else:
        residual_section = struct.pack('<I', 0) + residual_bitmap

    # Row scales
    row_scale_bytes = row_scales.cpu().float().numpy().tobytes()

    # Metadata
    meta = metadata or {}
    meta.update({
        'shape': list(prog.shape),
        'row_scale_shape': list(row_scales.shape),
    })
    meta_json = json.dumps(meta).encode()
    meta_len = struct.pack('<Q', len(meta_json))

    return (
        header
        + atom_bytes
        + coeff_bytes
        + atom_id_bytes
        + coeff_id_bytes
        + residual_section
        + row_scale_bytes
        + meta_len
        + meta_json
    )


def deserialize_wal_v2(data: bytes) -> Tuple[ProgramBufferV2, AtomTable, CoeffTable, torch.Tensor, Dict]:
    """Deserialize compact binary to WAL v2 state."""
    offset = 0

    # Header
    magic, version, K, C, flags, N, M, _, _ = struct.unpack_from('<4sHHHHQIB7s', data, offset)
    assert magic == MAGIC, f"Invalid magic: {magic}"
    assert version == VERSION, f"Unsupported version: {version}"
    offset += 32

    has_residuals = bool(flags & FLAG_HAS_RESIDUALS)
    packed_coeffs = bool(flags & FLAG_PACKED_COEFFS)

    # Atom table
    atoms_data = torch.frombuffer(data, dtype=torch.float32, count=K, offset=offset)
    offset += K * 4

    # Coeff table
    coeffs_data = torch.frombuffer(data, dtype=torch.float32, count=C, offset=offset)
    offset += C * 4

    # atom_ids
    atom_ids = torch.frombuffer(data, dtype=torch.uint8, count=N, offset=offset)
    offset += N

    # coeff_ids
    if packed_coeffs:
        packed_len = (N + 1) // 2
        coeff_packed = torch.frombuffer(data, dtype=torch.uint8, count=packed_len, offset=offset)
        coeff_ids = _unpack_uint4(coeff_packed, N)
        offset += packed_len
    else:
        coeff_ids = torch.frombuffer(data, dtype=torch.uint8, count=N, offset=offset)
        offset += N

    # Residuals
    residual_count = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    bitmap_len = (N + 7) // 8
    has_residual = _unpack_bitmap(data[offset:offset + bitmap_len], N)
    offset += bitmap_len

    residuals = torch.zeros(N, dtype=torch.float32)
    if residual_count > 0:
        residual_indices = torch.frombuffer(
            data, dtype=torch.uint32, count=residual_count, offset=offset
        ).long()
        offset += residual_count * 4

        residual_values = torch.frombuffer(
            data, dtype=torch.float16, count=residual_count, offset=offset
        ).float()
        offset += residual_count * 2

        residuals[residual_indices] = residual_values

    # Row scales
    row_scales = torch.frombuffer(data, dtype=torch.float32, count=M, offset=offset)
    offset += M * 4

    # Metadata
    meta_len = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    meta_json = json.loads(data[offset:offset + meta_len])

    # Reconstruct shape from metadata
    shape = tuple(meta_json['shape'])

    prog = ProgramBufferV2(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residuals,
        has_residual=has_residual,
        shape=shape,
    )
    atoms = AtomTable(atoms_data)
    coeffs = CoeffTable(coeffs_data)

    return prog, atoms, coeffs, row_scales, meta_json
