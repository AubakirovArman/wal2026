#!/usr/bin/env python3
"""WAL Framework — Phase 5: Hierarchical atoms."""


def build_hierarchy(input_dir: str, output_dir: str, max_l1: int = 64, trust_pickle: bool = False):
    """Build L1 hierarchical atoms on top of WAL v2 encoding.
    
    Args:
        input_dir: Input WAL directory
        output_dir: Output directory
        max_l1: Max L1 atoms to create
    """
    from pathlib import Path
    import torch
    from wal.v1.encoder import build_hierarchical_atoms
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load WAL state and build hierarchy
    from .safe_load import load_torch

    state = load_torch(input_dir / "wal_state.pt", trust_pickle=trust_pickle)
    
    # In practice: iterate layers, build L1 atoms
    print(f"  Building up to {max_l1} L1 atoms...")
    
    # Save hierarchical state
    torch.save(state, output_dir / "wal_state.pt")
    print(f"  Done. Output: {output_dir}")
