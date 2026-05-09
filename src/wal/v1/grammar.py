#!/usr/bin/env python3
"""WAL v1 Grammar — extended syntax for hierarchical atoms.

Text format:
  K 256
  C 16
  SHAPE 8192 8192
  
  ; Base atoms (L0)
  ATOM 0 = 0.123456
  ATOM 1 = -0.789012
  
  ; Composite atoms (L1)
  ATOM 256 = ADD(ATOM 5 * 0.3, ATOM 7 * 0.7)
  ATOM 257 = MUL(ATOM 12, ATOM 3)
  
  ; Programs
  ATOM 120 COEF 0.771360
  ATOM 256 COEF 0.500000  ; uses L1 composite atom
"""
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class WALAtomDef:
    """Parsed atom definition."""
    atom_id: int
    level: int
    op: str
    children: Optional[List[int]]
    coeffs: Optional[List[float]]
    value: Optional[float]  # for L0


@dataclass
class WALProgram:
    """Parsed program (same as v2)."""
    atom_id: int
    coeff_value: float
    residual: Optional[float] = None


def parse_wal(text: str) -> Tuple[List[WALAtomDef], List[WALProgram], int, int, Tuple[int, ...]]:
    """Parse WAL v1 text format."""
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith(';')]
    
    atom_defs = []
    programs = []
    K = None
    C = None
    shape = None
    
    for line in lines:
        if line.startswith('K '):
            K = int(line.split()[1])
        elif line.startswith('C '):
            C = int(line.split()[1])
        elif line.startswith('SHAPE '):
            shape = tuple(int(x) for x in line.split()[1:])
        elif line.startswith('ATOM ') and '=' in line:
            # Atom definition
            match = re.match(r'ATOM\s+(\d+)\s*=\s*(.+)', line)
            if match:
                atom_id = int(match.group(1))
                rhs = match.group(2).strip()
                
                if rhs.startswith('ADD(') or rhs.startswith('MUL('):
                    op = rhs[:3]
                    inner = rhs[4:-1]  # remove ADD( and )
                    parts = [p.strip() for p in inner.split(',')]
                    children = []
                    coeffs = []
                    for part in parts:
                        m = re.match(r'ATOM\s+(\d+)\s*\*\s*([\d.+-]+)', part)
                        if m:
                            children.append(int(m.group(1)))
                            coeffs.append(float(m.group(2)))
                        else:
                            m = re.match(r'ATOM\s+(\d+)', part)
                            if m:
                                children.append(int(m.group(1)))
                                coeffs.append(1.0)
                    atom_defs.append(WALAtomDef(atom_id, 1, op, children, coeffs, None))
                else:
                    # L0 scalar
                    value = float(rhs)
                    atom_defs.append(WALAtomDef(atom_id, 0, "CONST", None, None, value))
        elif line.startswith('ATOM ') and 'COEF' in line:
            # Program
            parts = line.split()
            atom_id = int(parts[1])
            coeff = float(parts[3])
            residual = float(parts[5]) if len(parts) > 5 else None
            programs.append(WALProgram(atom_id, coeff, residual))
    
    return atom_defs, programs, K, C, shape


def format_wal(atom_defs: List[WALAtomDef], programs: List[WALProgram],
               K: int, C: int, shape: Tuple[int, ...]) -> str:
    """Format WAL v1 text."""
    lines = [
        f"K {K}",
        f"C {C}",
        f"SHAPE {' '.join(map(str, shape))}",
        "",
        "; Atom Definitions",
    ]
    
    for d in atom_defs:
        if d.level == 0:
            lines.append(f"ATOM {d.atom_id} = {d.value:.6f}")
        else:
            args = ", ".join(f"ATOM {c} * {cf:.6f}" for c, cf in zip(d.children, d.coeffs))
            lines.append(f"ATOM {d.atom_id} = {d.op}({args})")
    
    lines.extend(["", "; Programs"])
    for p in programs:
        if p.residual is not None:
            lines.append(f"ATOM {p.atom_id} COEF {p.coeff_value:.6f} RESIDUAL {p.residual:.6f}")
        else:
            lines.append(f"ATOM {p.atom_id} COEF {p.coeff_value:.6f}")
    
    return "\n".join(lines)
