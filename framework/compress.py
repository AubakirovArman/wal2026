#!/usr/bin/env python3
"""WAL Framework — Phase 4: Binary compression."""


def compress_wal(input_dir: str, output_path: str, trust_pickle: bool = False):
    """Compress WAL directory to binary file.
    
    Args:
        input_dir: Input WAL directory
        output_path: Output binary file
    """
    import torch
    from pathlib import Path
    from wal.v1.format import serialize_wal_v1
    
    input_dir = Path(input_dir)
    
    # Load WAL state
    from .safe_load import load_torch

    state = load_torch(input_dir / "wal_state.pt", trust_pickle=trust_pickle)
    
    # Serialize each layer
    with open(output_path, "wb") as f:
        for name, blob in state.items():
            if isinstance(blob, bytes):
                f.write(blob)
    
    print(f"  Compressed to {output_path}")
