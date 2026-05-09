"""WAL v1 Binary Format: compact serialization with hierarchical atoms.

Format v1.0:
  Header (32 bytes):
    magic:       b'WAL1' (4 bytes)
    version:     uint16 (1)
    K0:          uint16 (base atoms count)
    K_total:     uint16 (total atoms including hierarchical)
    C:           uint16
    flags:       uint16 (bit 0: has_residuals, bit 1: packed_coeffs)
    N_weights:   uint64
    reserved:    10 bytes

  Base Atom Table: K0 × 4 bytes (float32)

  Hierarchical Atom Definitions:
    count: uint16 (K_total - K0)
    For each definition:
      op:         uint8 (0=ADD, 1=MUL, 2=NEG, 3=CONST)
      n_children: uint8
      children:   n_children × uint16
      coeffs:     n_children × 4 bytes (float32)

  Coeff Table: C × 4 bytes (float32)

  Programs:
    atom_ids:       N × 1 byte (uint8)
    coeff_ids:      ceil(N/2) bytes (uint4 packed) if FLAG_PACKED_COEFFS
                    OR N × 1 byte (uint8) if unpacked
    has_residual:   ceil(N/8) bytes (bitmap)
    residual_count: uint32
    residual_indices: residual_count × uint32 (if count > 0)
    residual_values:  residual_count × 2 bytes (float16, if count > 0)

  Metadata: JSON length-prefixed (uint64 length + JSON bytes)
"""
import json
import struct
import torch
import numpy as np
from typing import Dict, Tuple, List
from .isa import ProgramBufferV1, AtomTableV1, AtomDef, CoeffTable


MAGIC = b'WAL1'
VERSION = 1

# Flags
FLAG_HAS_RESIDUALS = 1 << 0
FLAG_PACKED_COEFFS = 1 << 1

# Op codes
OP_ADD = 0
OP_MUL = 1
OP_NEG = 2
OP_CONST = 3

_OP_ENCODE = {"ADD": OP_ADD, "MUL": OP_MUL, "NEG": OP_NEG, "CONST": OP_CONST}
_OP_DECODE = {v: k for k, v in _OP_ENCODE.items()}


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


def _serialize_hierarchical(atom_defs: List[AtomDef]) -> bytes:
    """Serialize hierarchical atom definitions to bytes.
    
    Only serializes L1+ definitions (index >= K0).
    Assumes atom_defs[0:K0] are L0 base atoms.
    """
    # Count L1+ defs
    l1_defs = [(i, d) for i, d in enumerate(atom_defs) if d.level > 0]
    count = len(l1_defs)
    
    buf = struct.pack('<H', count)
    for idx, d in l1_defs:
        op_code = _OP_ENCODE.get(d.op, OP_CONST)
        n_children = len(d.children) if d.children else 0
        buf += struct.pack('<BB', op_code, n_children)
        if n_children > 0:
            buf += struct.pack('<' + 'H' * n_children, *d.children)
            buf += struct.pack('<' + 'f' * n_children, *d.coeffs)
    return buf


def _deserialize_hierarchical(data: bytes, offset: int, K0: int) -> Tuple[List[AtomDef], int]:
    """Deserialize hierarchical atom definitions from bytes.
    
    Returns (atom_defs_list, new_offset).
    """
    count = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    
    defs = [AtomDef(level=0, op="CONST") for _ in range(K0)]
    
    for _ in range(count):
        op_code, n_children = struct.unpack_from('<BB', data, offset)
        offset += 2
        op = _OP_DECODE.get(op_code, "CONST")
        
        if n_children > 0:
            children = list(struct.unpack_from('<' + 'H' * n_children, data, offset))
            offset += n_children * 2
            coeffs = list(struct.unpack_from('<' + 'f' * n_children, data, offset))
            offset += n_children * 4
        else:
            children = None
            coeffs = None
        
        level = 1 if op_code != OP_CONST else 0
        defs.append(AtomDef(level=level, op=op, children=children, coeffs=coeffs))
    
    return defs, offset


def serialize_wal_v1(
    prog: ProgramBufferV1,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    metadata: Dict = None,
) -> bytes:
    """Serialize WAL v1 state to compact binary.
    
    Args:
        prog: Program buffer
        atom_table: Hierarchical atom table
        coeffs: Coefficient table
        metadata: Optional metadata dict
    
    Returns:
        Binary blob as bytes
    """
    K0 = atom_table.K0
    K_total = atom_table.K_total
    C = coeffs.values.numel()
    N = prog.N
    
    has_residuals = prog.has_residual.any().item()
    flags = FLAG_PACKED_COEFFS
    if has_residuals:
        flags |= FLAG_HAS_RESIDUALS
    
    # Header (32 bytes)
    header = struct.pack(
        '<4sHHHHHQ10s',
        MAGIC, VERSION, K0, K_total, C, flags, N, b'\x00' * 10,
    )
    assert len(header) == 32
    
    # Base atom table
    atom_bytes = atom_table.base_atoms.cpu().float().numpy().tobytes()
    
    # Hierarchical definitions
    hier_bytes = _serialize_hierarchical(atom_table.atom_defs)
    
    # Coeff table
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
    
    # Metadata
    meta = metadata or {}
    meta.update({
        'shape': list(prog.shape),
    })
    meta_json = json.dumps(meta).encode()
    meta_len = struct.pack('<Q', len(meta_json))
    
    return (
        header
        + atom_bytes
        + hier_bytes
        + coeff_bytes
        + atom_id_bytes
        + coeff_id_bytes
        + residual_section
        + meta_len
        + meta_json
    )


def deserialize_wal_v1(data: bytes) -> Tuple[ProgramBufferV1, AtomTableV1, CoeffTable, Dict]:
    """Deserialize compact binary to WAL v1 state.
    
    Args:
        data: Binary blob
    
    Returns:
        (prog, atom_table, coeffs, metadata)
    """
    offset = 0
    
    # Header
    magic, version, K0, K_total, C, flags, N = struct.unpack_from('<4sHHHHHQ', data, offset)
    assert magic == MAGIC, f"Invalid magic: {magic}"
    assert version == VERSION, f"Unsupported version: {version}"
    offset += 32
    
    has_residuals = bool(flags & FLAG_HAS_RESIDUALS)
    packed_coeffs = bool(flags & FLAG_PACKED_COEFFS)
    
    # Base atom table
    base_atoms = torch.frombuffer(data, dtype=torch.float32, count=K0, offset=offset)
    offset += K0 * 4
    
    # Hierarchical definitions
    atom_defs, offset = _deserialize_hierarchical(data, offset, K0)
    assert len(atom_defs) == K_total, f"Expected {K_total} defs, got {len(atom_defs)}"
    
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
    
    # Metadata
    meta_len = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    meta_json = json.loads(data[offset:offset + meta_len])
    
    # Reconstruct
    shape = tuple(meta_json['shape'])
    
    prog = ProgramBufferV1(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residuals,
        has_residual=has_residual,
        shape=shape,
    )
    atom_table = AtomTableV1(base_atoms=base_atoms, atom_defs=atom_defs)
    coeffs = CoeffTable(values=coeffs_data)
    
    return prog, atom_table, coeffs, meta_json
