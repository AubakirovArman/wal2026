#!/usr/bin/env python3
"""WAL Framework — Phase 2: Grammar conversions."""


def convert_format(input_path: str, output_path: str, target_format: str = "text"):
    """Convert between WAL text and binary formats.
    
    Args:
        input_path: Input file
        output_path: Output file
        target_format: 'text' or 'binary'
    """
    import torch
    from wal.v2.grammar import parse_wal_text, format_wal_text
    from wal.v2.format import serialize_wal_v2, deserialize_wal_v2
    
    if target_format == "text":
        # Binary → Text
        with open(input_path, "rb") as f:
            data = f.read()
        prog, atoms, coeffs, _row_scales, _meta = deserialize_wal_v2(data)
        text = format_wal_text(prog, atoms, coeffs)
        with open(output_path, "w") as f:
            f.write(text)
    else:
        # Text → Binary
        with open(input_path, "r") as f:
            text = f.read()
        prog, atoms, coeffs = parse_wal_text(text)
        rows = int(prog.shape[0]) if prog.shape else 1
        row_scales = torch.ones(rows, dtype=torch.float32)
        data = serialize_wal_v2(prog, atoms, coeffs, row_scales)
        with open(output_path, "wb") as f:
            f.write(data)
