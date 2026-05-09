#!/usr/bin/env python3
"""WAL v1 ISA — Hierarchical Atoms.

Extends WAL v2 with hierarchical atom definitions.
An atom can be a base scalar (L0) or a composite (L1+) defined as a tree of operations.
Programs remain simple: atom_id + coeff_id + residual.
But atoms themselves have structure.
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch


@dataclass 
class AtomDef:
    """Definition of a hierarchical atom."""
    level: int  # 0 = base scalar, 1 = composite, etc.
    # For L0: value is the scalar itself (stored in AtomTable)
    # For L1+: operation tree
    op: str = "CONST"  # "CONST", "ADD", "MUL", "NEG"
    children: Optional[List[int]] = None  # child atom_ids in AtomTable
    coeffs: Optional[List[float]] = None  # coefficients for children
    
    def is_leaf(self) -> bool:
        return self.level == 0


@dataclass
class AtomTableV1:
    """Hierarchical atom table with metadata."""
    base_atoms: torch.Tensor  # [K0] float32 — L0 scalar values
    atom_defs: List[AtomDef]  # [K_total] definitions for all levels
    
    @property
    def K0(self) -> int:
        return self.base_atoms.numel()
    
    @property
    def K_total(self) -> int:
        return len(self.atom_defs)
    
    def resolve(self, atom_id: int) -> float:
        """Recursively resolve atom to scalar value."""
        def _resolve(aid: int) -> float:
            d = self.atom_defs[aid]
            if d.is_leaf():
                return self.base_atoms[aid].item()
            vals = [_resolve(c) for c in d.children]
            if d.op == "ADD":
                return sum(v * c for v, c in zip(vals, d.coeffs))
            elif d.op == "MUL":
                result = vals[0]
                for v, c in zip(vals[1:], d.coeffs[1:]):
                    result *= v * c
                return result
            elif d.op == "NEG":
                return -vals[0]
            else:
                raise ValueError(f"Unknown op: {d.op}")
        return _resolve(atom_id)


@dataclass
class ProgramBufferV1:
    """WAL v1 program buffer. Same structure as v2, but atoms are hierarchical."""
    atom_ids: torch.Tensor      # [N] uint8
    coeff_ids: torch.Tensor     # [N] uint8
    residuals: torch.Tensor     # [N] float16 or empty
    has_residual: torch.Tensor  # [N] bool
    shape: Tuple[int, ...]
    
    @property
    def N(self) -> int:
        return self.atom_ids.numel()


@dataclass
class CoeffTable:
    """Coefficient table (same as v2)."""
    values: torch.Tensor  # [C] float32
