"""WAL v2 Assembler/Disassembler: text <-> binary.

Assembler:   WAL text + AtomTable + CoeffTable -> ProgramBufferV2
Disassembler: ProgramBufferV2 + AtomTable + CoeffTable -> WAL text
"""
import torch
from typing import Tuple
from .isa import ProgramBufferV2, AtomTable, CoeffTable, WALProgram
from .grammar import (
    parse_program_stream, format_program_stream, format_unique_programs,
    ParsedProgram, WALParseError,
)


def assemble(
    text: str,
    atoms: AtomTable,
    coeffs: CoeffTable,
    expected_shape: Tuple[int, ...] = None,
) -> ProgramBufferV2:
    """Assemble WAL text to ProgramBufferV2.
    
    Validates that parsed K/C match tables, and that number of programs
    matches expected shape. Quantizes coeff_values to nearest coeff_id.
    
    Args:
        text: WAL v2 text format
        atoms: Atom table (for validation)
        coeffs: Coeff table (for quantization)
        expected_shape: Expected matrix shape (optional, for validation)
        
    Returns:
        ProgramBufferV2
        
    Raises:
        WALParseError: on syntax error
        ValueError: on validation error
    """
    parsed = parse_program_stream(text)
    
    # Validate header
    if parsed.K != atoms.K:
        raise ValueError(f"Text says K={parsed.K}, but atom table has K={atoms.K}")
    if parsed.C != coeffs.C:
        raise ValueError(f"Text says C={parsed.C}, but coeff table has C={coeffs.C}")
    
    N = len(parsed.programs)
    
    if expected_shape is not None:
        if parsed.shape != expected_shape:
            raise ValueError(
                f"Text says SHAPE={parsed.shape}, but expected {expected_shape}"
            )
        expected_N = expected_shape[0] * expected_shape[1]
        if N != expected_N:
            raise ValueError(
                f"Text has {N} programs, but shape {expected_shape} expects {expected_N}"
            )
    else:
        expected_shape = parsed.shape
    
    device = atoms.values.device
    
    atom_ids = torch.empty(N, dtype=torch.uint8, device='cpu')
    coeff_ids = torch.empty(N, dtype=torch.uint8, device='cpu')
    residuals = torch.zeros(N, dtype=torch.float32, device='cpu')
    has_residual = torch.zeros(N, dtype=torch.bool, device='cpu')
    
    coeff_values = coeffs.values.cpu().numpy()
    
    for i, prog in enumerate(parsed.programs):
        # Validate atom_id
        if not (0 <= prog.atom_id < atoms.K):
            raise ValueError(
                f"Program {i}: atom_id={prog.atom_id} out of range [0, {atoms.K})"
            )
        atom_ids[i] = prog.atom_id
        
        # Quantize coeff_value to nearest coeff_id (absolute difference)
        import numpy as np
        coeff_id = int(np.abs(coeff_values - prog.coeff_value).argmin())
        if not (0 <= coeff_id < coeffs.C):
            raise ValueError(
                f"Program {i}: coeff_value={prog.coeff_value} quantizes to "
                f"coeff_id={coeff_id} out of range [0, {coeffs.C})"
            )
        coeff_ids[i] = coeff_id
        
        # Store residual if present
        if prog.residual is not None:
            residuals[i] = prog.residual
            has_residual[i] = True
    
    return ProgramBufferV2(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residuals,
        has_residual=has_residual,
        shape=expected_shape,
    )


def disassemble(
    prog: ProgramBufferV2,
    atoms: AtomTable,
    coeffs: CoeffTable,
    max_programs: int = 100,
    format: str = "full",
) -> str:
    """Disassemble ProgramBufferV2 to WAL text.
    
    Args:
        prog: Program buffer
        atoms: Atom table (for coeff value lookup)
        coeffs: Coeff table (for coeff value lookup)
        max_programs: Max programs to output (None = all)
        format: "full" or "unique"
        
    Returns:
        WAL v2 text
    """
    coeff_values = coeffs.values.cpu().numpy()
    
    if format == "unique":
        # For unique format, we need all programs to count them
        # But we can do this efficiently without building full list
        from collections import Counter
        keys = []
        for i in range(prog.N):
            atom_id = int(prog.atom_ids[i].item())
            coeff_id = int(prog.coeff_ids[i].item())
            coeff_val = float(coeff_values[coeff_id])
            residual = None
            if prog.has_residual[i]:
                residual = float(prog.residuals[i].item())
            keys.append((atom_id, coeff_val, residual))
        
        counts = Counter(keys)
        programs = [ParsedProgram(k[0], k[1], k[2]) for k in counts.keys()]
        
        # Build text manually with counts
        lines = [
            f"; WAL v2 v0.1 — Unique Program Summary",
            f"; K={atoms.K} C={coeffs.C} SHAPE={'x'.join(map(str, prog.shape))}",
            f"; {len(programs)} unique / {prog.N} total weights",
            f"K {atoms.K}",
            f"C {coeffs.C}",
            f"SHAPE {prog.shape[0]} {prog.shape[1]}",
            "",
            "; <count> | <program>",
        ]
        for key, count in counts.most_common():
            atom_id, coeff_val, residual = key
            line = f"  {count:>10} | ATOM {atom_id} COEF {coeff_val:.6f}"
            if residual is not None:
                line += f" RESIDUAL {residual:.8f}"
            lines.append(line)
        return '\n'.join(lines)
    
    else:
        # Full format: only output up to max_programs
        total = prog.N
        limit = max_programs if max_programs is not None else total
        
        lines = [
            f"; WAL v2 v0.1 — {total:,} programs",
            f"K {atoms.K}",
            f"C {coeffs.C}",
            f"SHAPE {prog.shape[0]} {prog.shape[1]}",
            "",
        ]
        
        for i in range(min(limit, total)):
            atom_id = int(prog.atom_ids[i].item())
            coeff_id = int(prog.coeff_ids[i].item())
            coeff_val = float(coeff_values[coeff_id])
            line = f"ATOM {atom_id} COEF {coeff_val:.6f}"
            if prog.has_residual[i]:
                residual = float(prog.residuals[i].item())
                line += f" RESIDUAL {residual:.8f}"
            lines.append(line)
        
        if total > limit:
            lines.append(f"; ... {total - limit} more programs ...")
        
        return '\n'.join(lines)


def disassemble_single(prog: WALProgram, atoms: AtomTable, coeffs: CoeffTable) -> str:
    """Disassemble a single WAL program to text line.
    
    Args:
        prog: Single program
        atoms: Atom table
        coeffs: Coeff table
        
    Returns:
        Text representation
    """
    coeff_val = float(coeffs.values[prog.coeff_id].item())
    line = f"ATOM {prog.atom_id} COEF {coeff_val:.6f}"
    if prog.has_residual:
        line += f" RESIDUAL {prog.residual:.8f}"
    return line
