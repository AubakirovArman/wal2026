#!/usr/bin/env python3
"""WAL v1 Assembler / Disassembler.

Converts between text format and ProgramBufferV1 + AtomTableV1.
"""
import torch
from typing import Tuple
from .isa import AtomTableV1, AtomDef, ProgramBufferV1, CoeffTable
from .grammar import parse_wal, format_wal, WALAtomDef, WALProgram


def assemble(text: str, device=None) -> Tuple[ProgramBufferV1, AtomTableV1, CoeffTable, dict]:
    """Assemble WAL v1 text into binary structures."""
    atom_defs, programs, K, C, shape = parse_wal(text)
    device = device or torch.device("cpu")
    
    # Build atom table
    base_atoms = torch.zeros(K, device=device, dtype=torch.float32)
    defs = []
    for d in atom_defs:
        if d.level == 0:
            base_atoms[d.atom_id] = d.value
            defs.append(AtomDef(level=0, op="CONST"))
        else:
            defs.append(AtomDef(
                level=d.level,
                op=d.op,
                children=d.children,
                coeffs=d.coeffs,
            ))
    
    atom_table = AtomTableV1(base_atoms=base_atoms, atom_defs=defs)
    
    # Build coeff table (uniform for now, can be learned)
    coeff_table = CoeffTable(values=torch.linspace(0.1, 2.0, C, device=device))
    
    # Build program buffer
    N = len(programs)
    atom_ids = torch.empty(N, dtype=torch.uint8, device=device)
    coeff_ids = torch.empty(N, dtype=torch.uint8, device=device)
    
    for i, p in enumerate(programs):
        atom_ids[i] = p.atom_id
        # Find nearest coeff
        c_idx = (coeff_table.values - p.coeff_value).abs().argmin()
        coeff_ids[i] = c_idx
    
    prog = ProgramBufferV1(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=torch.empty(0, dtype=torch.float16, device=device),
        has_residual=torch.zeros(N, dtype=torch.bool, device=device),
        shape=shape,
    )
    
    meta = {"K": K, "C": C}
    return prog, atom_table, coeff_table, meta


def disassemble(prog: ProgramBufferV1, atom_table: AtomTableV1, 
                coeffs: CoeffTable, include_defs: bool = True) -> str:
    """Disassemble binary structures into WAL v1 text."""
    N = prog.N
    shape = prog.shape
    K = atom_table.K0
    C = coeffs.values.numel()
    
    # Build atom definitions
    atom_defs = []
    if include_defs:
        for i in range(K):
            atom_defs.append(WALAtomDef(i, 0, "CONST", None, None, atom_table.base_atoms[i].item()))
        for i in range(K, atom_table.K_total):
            d = atom_table.atom_defs[i]
            atom_defs.append(WALAtomDef(i, d.level, d.op, d.children, d.coeffs, None))
    
    # Build programs
    programs = []
    flat = prog.atom_ids.reshape(-1)
    c_flat = prog.coeff_ids.reshape(-1)
    for i in range(N):
        programs.append(WALProgram(
            atom_id=int(flat[i]),
            coeff_value=float(coeffs.values[int(c_flat[i])]),
        ))
    
    return format_wal(atom_defs, programs, K, C, shape)
