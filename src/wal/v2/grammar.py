"""WAL v2 Grammar: BNF definition, parser, and pretty-printer.

Text Format v0.1:
  ; Comment line
  K <int>                      ; atom table size
  C <int>                      ; coeff table size
  SHAPE <int> <int>            ; matrix shape (rows cols)
  ATOM <id> COEF <value>       ; atom call with coefficient
  ATOM <id> COEF <value> RESIDUAL <value>  ; with explicit residual

Example:
  K 256
  C 16
  SHAPE 8192 8192
  ATOM 42 COEF 1.5000
  ATOM 7 COEF -0.2500
  ATOM 0 COEF 0.0000 RESIDUAL 0.001200
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch

from .isa import AtomTable, CoeffTable, ProgramBufferV2


# Formal BNF for WAL v2 text format
WAL_BNF = """
<program_stream>   ::= <header> <program>*
<header>           ::= <k_decl> <c_decl> <shape_decl>
<k_decl>           ::= "K" <uint>
<c_decl>           ::= "C" <uint>
<shape_decl>       ::= "SHAPE" <uint> <uint>
<program>          ::= <atom_call> [<residual>]
<atom_call>        ::= "ATOM" <atom_id> "COEF" <float>
<residual>         ::= "RESIDUAL" <float>
<atom_id>          ::= <uint> [0, K-1]
<coeff_value>      ::= <float>
<uint>             ::= digit+
<float>            ::= ["-" | "+"] digit+ ["." digit+]
<comment>          ::= ";" <any>* <newline>
"""


@dataclass
class ParsedProgram:
    """Single parsed WAL v2 program."""
    atom_id: int
    coeff_value: float
    residual: Optional[float] = None


@dataclass
class ParsedStream:
    """Parsed WAL v2 program stream."""
    K: int
    C: int
    shape: Tuple[int, int]
    programs: List[ParsedProgram]


class WALParseError(Exception):
    """Raised when WAL text cannot be parsed."""
    pass


def _tokenize_line(line: str) -> List[str]:
    """Strip comments and split line into tokens."""
    # Remove comments
    if ';' in line:
        line = line[:line.index(';')]
    line = line.strip()
    if not line:
        return []
    return line.split()


def parse_program_stream(text: str) -> ParsedStream:
    """Parse WAL v2 text format to structured representation.
    
    Args:
        text: Multi-line WAL text
        
    Returns:
        ParsedStream with header and programs
        
    Raises:
        WALParseError: on syntax error
    """
    lines = text.strip().split('\n')
    
    K: Optional[int] = None
    C: Optional[int] = None
    shape: Optional[Tuple[int, int]] = None
    programs: List[ParsedProgram] = []
    
    for line_no, raw_line in enumerate(lines, 1):
        tokens = _tokenize_line(raw_line)
        if not tokens:
            continue
        
        if tokens[0] == 'K':
            if len(tokens) != 2:
                raise WALParseError(f"Line {line_no}: K declaration expects 1 argument")
            K = int(tokens[1])
            
        elif tokens[0] == 'C':
            if len(tokens) != 2:
                raise WALParseError(f"Line {line_no}: C declaration expects 1 argument")
            C = int(tokens[1])
            
        elif tokens[0] == 'SHAPE':
            if len(tokens) != 3:
                raise WALParseError(f"Line {line_no}: SHAPE expects 2 arguments")
            shape = (int(tokens[1]), int(tokens[2]))
            
        elif tokens[0] == 'ATOM':
            # ATOM <id> COEF <value> [RESIDUAL <value>]
            if len(tokens) not in (4, 6):
                raise WALParseError(
                    f"Line {line_no}: ATOM expects 'ATOM <id> COEF <value>' "
                    f"or 'ATOM <id> COEF <value> RESIDUAL <value>'"
                )
            if tokens[2] != 'COEF':
                raise WALParseError(f"Line {line_no}: Expected 'COEF' after atom id")
            
            atom_id = int(tokens[1])
            coeff_value = float(tokens[3])
            residual = None
            
            if len(tokens) == 6:
                if tokens[4] != 'RESIDUAL':
                    raise WALParseError(f"Line {line_no}: Expected 'RESIDUAL' after coeff value")
                residual = float(tokens[5])
            
            programs.append(ParsedProgram(atom_id, coeff_value, residual))
            
        else:
            raise WALParseError(f"Line {line_no}: Unknown directive '{tokens[0]}'")
    
    if K is None:
        raise WALParseError("Missing K declaration")
    if C is None:
        raise WALParseError("Missing C declaration")
    if shape is None:
        raise WALParseError("Missing SHAPE declaration")
    
    return ParsedStream(K=K, C=C, shape=shape, programs=programs)


def format_program_stream(
    programs: List[ParsedProgram],
    shape: Tuple[int, int],
    K: int,
    C: int,
    max_programs: Optional[int] = None,
) -> str:
    """Format programs to WAL v2 text.
    
    Args:
        programs: List of parsed programs
        shape: Matrix shape
        K: Atom table size
        C: Coeff table size
        max_programs: If set, only output first N programs + ellipsis
        
    Returns:
        WAL text
    """
    lines = [
        f"; WAL v2 v0.1 — {len(programs):,} programs",
        f"K {K}",
        f"C {C}",
        f"SHAPE {shape[0]} {shape[1]}",
        "",
    ]
    
    total = len(programs)
    limit = max_programs if max_programs is not None else total
    
    for i in range(min(limit, total)):
        p = programs[i]
        line = f"ATOM {p.atom_id} COEF {p.coeff_value:.6f}"
        if p.residual is not None:
            line += f" RESIDUAL {p.residual:.8f}"
        lines.append(line)
    
    if total > limit:
        lines.append(f"; ... {total - limit} more programs ...")
    
    return '\n'.join(lines)


def format_unique_programs(
    programs: List[ParsedProgram],
    shape: Tuple[int, int],
    K: int,
    C: int,
) -> str:
    """Format unique programs with occurrence counts (summary view).
    
    Args:
        programs: List of parsed programs
        shape: Matrix shape
        K: Atom table size
        C: Coeff table size
        
    Returns:
        WAL text with unique programs and counts
    """
    from collections import Counter
    
    # Create hashable keys
    keys = []
    for p in programs:
        key = (p.atom_id, p.coeff_value, p.residual)
        keys.append(key)
    
    counts = Counter(keys)
    unique = list(counts.keys())
    
    lines = [
        f"; WAL v2 v0.1 — Unique Program Summary",
        f"; K={K} C={C} SHAPE={'x'.join(map(str, shape))}",
        f"; {len(unique)} unique / {len(programs)} total weights",
        f"K {K}",
        f"C {C}",
        f"SHAPE {shape[0]} {shape[1]}",
        "",
        "; <count> | <program>",
    ]
    
    # Sort by count descending
    for key, count in counts.most_common():
        atom_id, coeff_val, residual = key
        line = f"  {count:>10} | ATOM {atom_id} COEF {coeff_val:.6f}"
        if residual is not None:
            line += f" RESIDUAL {residual:.8f}"
        lines.append(line)
    
    return '\n'.join(lines)


def format_wal_text(
    prog: ProgramBufferV2,
    atoms: AtomTable,
    coeffs: CoeffTable,
    max_programs: Optional[int] = None,
) -> str:
    """Format a complete WAL v2 text file including atom/coeff tables.

    The lower-level grammar intentionally models only program streams. This
    helper is the public round-trip format used by CLI tools, where binary
    reconstruction also needs the numeric tables.
    """
    coeff_values = coeffs.values.detach().cpu()
    atom_values = atoms.values.detach().cpu()
    limit = prog.N if max_programs is None else min(int(max_programs), prog.N)
    lines = [
        f"; WAL v2 text — {prog.N:,} programs",
        f"K {atoms.K}",
        f"C {coeffs.C}",
        f"SHAPE {' '.join(str(int(x)) for x in prog.shape)}",
        "",
        "; Atom table",
    ]
    lines.extend(f"TABLE ATOM {idx} {float(value):.9g}" for idx, value in enumerate(atom_values))
    lines.append("")
    lines.append("; Coeff table")
    lines.extend(f"TABLE COEFF {idx} {float(value):.9g}" for idx, value in enumerate(coeff_values))
    lines.append("")
    lines.append("; Programs")
    for idx in range(limit):
        atom_id = int(prog.atom_ids[idx].item())
        coeff_id = int(prog.coeff_ids[idx].item())
        line = f"ATOM {atom_id} COEF {float(coeff_values[coeff_id]):.9g}"
        if bool(prog.has_residual[idx].item()):
            line += f" RESIDUAL {float(prog.residuals[idx].item()):.9g}"
        lines.append(line)
    if limit < prog.N:
        lines.append(f"; ... {prog.N - limit} more programs ...")
    return "\n".join(lines)


def parse_wal_text(text: str) -> Tuple[ProgramBufferV2, AtomTable, CoeffTable]:
    """Parse complete WAL v2 text produced by :func:`format_wal_text`.

    For legacy program-only text, table values are synthesized as zeros for
    atoms and sorted observed coefficient values for coefficients.
    """
    atom_values: dict[int, float] = {}
    coeff_values: dict[int, float] = {}
    program_lines: list[str] = []
    header_lines: list[str] = []

    for raw_line in text.splitlines():
        tokens = _tokenize_line(raw_line)
        if not tokens:
            continue
        if tokens[0] in {"K", "C", "SHAPE"}:
            header_lines.append(" ".join(tokens))
        elif len(tokens) == 4 and tokens[0] == "TABLE" and tokens[1] == "ATOM":
            atom_values[int(tokens[2])] = float(tokens[3])
        elif len(tokens) == 4 and tokens[0] == "TABLE" and tokens[1] == "COEFF":
            coeff_values[int(tokens[2])] = float(tokens[3])
        elif tokens[0] == "ATOM":
            program_lines.append(" ".join(tokens))
        elif tokens[0] == "TABLE":
            raise WALParseError(f"Unknown TABLE directive: {' '.join(tokens)}")

    parsed = parse_program_stream("\n".join(header_lines + program_lines))
    N = len(parsed.programs)

    if atom_values:
        atoms_tensor = torch.zeros(parsed.K, dtype=torch.float32)
        for idx, value in atom_values.items():
            if not 0 <= idx < parsed.K:
                raise WALParseError(f"Atom table index {idx} out of range [0, {parsed.K})")
            atoms_tensor[idx] = value
    else:
        atoms_tensor = torch.zeros(parsed.K, dtype=torch.float32)

    if coeff_values:
        coeffs_tensor = torch.zeros(parsed.C, dtype=torch.float32)
        for idx, value in coeff_values.items():
            if not 0 <= idx < parsed.C:
                raise WALParseError(f"Coeff table index {idx} out of range [0, {parsed.C})")
            coeffs_tensor[idx] = value
    else:
        observed = sorted({p.coeff_value for p in parsed.programs})
        coeffs_tensor = torch.zeros(parsed.C, dtype=torch.float32)
        for idx, value in enumerate(observed[: parsed.C]):
            coeffs_tensor[idx] = value

    atom_ids = torch.empty(N, dtype=torch.uint8)
    coeff_ids = torch.empty(N, dtype=torch.uint8)
    residuals = torch.zeros(N, dtype=torch.float32)
    has_residual = torch.zeros(N, dtype=torch.bool)
    for idx, program in enumerate(parsed.programs):
        if not 0 <= program.atom_id < parsed.K:
            raise WALParseError(f"Program {idx}: atom_id={program.atom_id} out of range")
        atom_ids[idx] = program.atom_id
        coeff_id = int(torch.argmin((coeffs_tensor - float(program.coeff_value)).abs()).item())
        coeff_ids[idx] = coeff_id
        if program.residual is not None:
            residuals[idx] = float(program.residual)
            has_residual[idx] = True

    prog = ProgramBufferV2(atom_ids, coeff_ids, residuals, has_residual, parsed.shape)
    return prog, AtomTable(atoms_tensor), CoeffTable(coeffs_tensor)
