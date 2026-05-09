"""WAL Binary Format: serialization for model shipping.

Format v0.1:
  Header (32 bytes):
    magic:      b'WAL0' (4 bytes)
    version:    uint16 (1)
    K:          uint16
    lmax:       uint16
    N_weights:  uint64  (total weights encoded)
    dtype:      uint8   (0=fp32, 1=fp16, 2=bf16)
    reserved:   9 bytes
  
  Atom Table: K * sizeof(dtype) bytes
  Programs:   N_weights * lmax * 2 bytes (uint8 idx + int8 sign)
  Metadata:   JSON with per-parameter shapes, devices, row_scales
"""
import json
import struct
import torch
from dataclasses import dataclass, asdict
from typing import Dict, List
from .isa import ProgramBuffer


MAGIC = b'WAL0'
VERSION = 1


@dataclass
class WALParameterMeta:
    """Metadata for one encoded parameter."""
    name: str
    shape: List[int]
    device: str
    row_scale_shape: List[int]
    offset: int  # offset in programs array
    numel: int
    is_encoded: bool


@dataclass
class WALModelState:
    """Complete WAL-encoded model state."""
    K: int
    lmax: int
    dtype_str: str  # 'float32', 'bfloat16', etc.
    atom_table: torch.Tensor  # [K] fp32
    programs: ProgramBuffer   # global program buffer for all encoded weights
    params: List[WALParameterMeta]
    
    def total_bytes(self) -> int:
        """Estimate compressed size."""
        header = 32
        atoms = self.atom_table.numel() * 4  # fp32
        prog_idx = self.programs.indices.numel() * 1  # uint8
        prog_sign = self.programs.signs.numel() * 1  # int8
        meta = len(json.dumps([asdict(p) for p in self.params]).encode())
        return header + atoms + prog_idx + prog_sign + meta
    
    def compression_ratio(self, original_bytes: int) -> float:
        return original_bytes / self.total_bytes()


def _dtype_to_code(dt: torch.dtype) -> int:
    mapping = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    return mapping.get(dt, 0)


def _code_to_dtype(code: int) -> torch.dtype:
    mapping = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}
    return mapping.get(code, torch.float32)


def serialize_wal_state(state: WALModelState) -> bytes:
    """Serialize WALModelState to bytes."""
    # Header
    N = state.programs.N
    dtype_code = _dtype_to_code(getattr(torch, state.dtype_str))
    header = struct.pack(
        '<4sHHHQH12s',
        MAGIC,
        VERSION,
        state.K,
        state.lmax,
        N,
        dtype_code,
        b'\x00' * 12,
    )
    assert len(header) == 32
    
    # Atom table (fp32)
    atoms_bytes = state.atom_table.cpu().float().numpy().tobytes()
    
    # Programs (uint8 indices + int8 signs)
    idx_bytes = state.programs.indices.cpu().numpy().tobytes()
    sign_bytes = state.programs.signs.cpu().numpy().tobytes()
    
    # Metadata JSON
    meta = {
        'K': state.K,
        'lmax': state.lmax,
        'dtype_str': state.dtype_str,
        'params': [asdict(p) for p in state.params],
    }
    meta_bytes = json.dumps(meta).encode()
    meta_len = struct.pack('<Q', len(meta_bytes))
    
    return header + atoms_bytes + idx_bytes + sign_bytes + meta_len + meta_bytes


def deserialize_wal_state(data: bytes) -> WALModelState:
    """Deserialize bytes to WALModelState."""
    offset = 0
    
    # Header
    magic, version, K, lmax, N, dtype_code, _ = struct.unpack_from('<4sHHHQH12s', data, offset)
    assert magic == MAGIC, f"Invalid magic: {magic}"
    assert version == VERSION, f"Unsupported version: {version}"
    offset += 32
    
    dtype = _code_to_dtype(dtype_code)
    dtype_str = str(dtype).split('.')[-1]
    
    # Atom table
    atom_size = K * 4
    atoms = torch.frombuffer(data, dtype=torch.float32, count=K, offset=offset)
    offset += atom_size
    
    # Programs
    indices = torch.frombuffer(data, dtype=torch.uint8, count=N * lmax, offset=offset)
    indices = indices.reshape(N, lmax)
    offset += N * lmax
    
    signs = torch.frombuffer(data, dtype=torch.int8, count=N * lmax, offset=offset)
    signs = signs.reshape(N, lmax)
    offset += N * lmax
    
    programs = ProgramBuffer(indices, signs, lmax)
    
    # Metadata
    meta_len = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    meta_json = json.loads(data[offset:offset + meta_len])
    
    params = [WALParameterMeta(**p) for p in meta_json['params']]
    
    return WALModelState(
        K=K,
        lmax=lmax,
        dtype_str=dtype_str,
        atom_table=atoms,
        programs=programs,
        params=params,
    )
