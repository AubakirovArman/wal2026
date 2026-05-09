#!/usr/bin/env python3
"""WAL v1 — Hierarchical Atoms.

Phase 5: Extends WAL v2 with hierarchical atom definitions.
Programs remain 12 bits/weight (atom_id + coeff_id), preserving WAL v2 PPL quality.
Atoms gain structure: L0 scalars, L1+ composites (ADD, MUL, etc.).
"""
from .isa import AtomTableV1, AtomDef, ProgramBufferV1, CoeffTable
from .encoder import build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms
from .decoder import wal_decode_v1, precompute_flat_atoms, apply_row_scale
from .grammar import parse_wal, format_wal, WALAtomDef, WALProgram
from .asm import assemble, disassemble
from .format import serialize_wal_v1, deserialize_wal_v1
from .nn import WALParameter, WALLinear, WALCachedLinear, encode_linear_weight, replace_linear_with_wal
from .runtime import WALModel
from .debugger import WALDebugger, Breakpoint, TraceRecord, HeatmapStats
from .stdlib import (
    AtomLibraryEntry, AtomLibrary,
    build_entry_from_encoded, create_default_library,
    encode_with_pretrained_atoms, evaluate_transfer, transfer_atoms_direct,
)

__all__ = [
    "AtomTableV1", "AtomDef", "ProgramBufferV1", "CoeffTable",
    "build_l0_atoms", "build_coeff_table", "wal_encode_v1", "build_hierarchical_atoms",
    "wal_decode_v1", "precompute_flat_atoms", "apply_row_scale",
    "parse_wal", "format_wal", "WALAtomDef", "WALProgram",
    "assemble", "disassemble",
    "serialize_wal_v1", "deserialize_wal_v1",
    "WALParameter", "WALLinear", "WALCachedLinear",
    "encode_linear_weight", "replace_linear_with_wal",
    "WALModel",
    "WALDebugger", "Breakpoint", "TraceRecord", "HeatmapStats",
    "AtomLibraryEntry", "AtomLibrary",
    "build_entry_from_encoded", "create_default_library",
    "encode_with_pretrained_atoms", "evaluate_transfer", "transfer_atoms_direct",
]
