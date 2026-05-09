#!/usr/bin/env python3
"""WAL Framework — Phase 6: PyTorch integration."""
import torch


def replace_linear(
    input_path: str,
    output_path: str,
    K: int = 256,
    C: int = 16,
    cached: bool = False,
    trust_pickle: bool = False,
):
    """Replace all nn.Linear in a model with WAL layers.
    
    Args:
        input_path: Input model path
        output_path: Output model path
        K: Number of atoms
        C: Number of coefficients
        cached: Use WALCachedLinear
    """
    from wal.v1.nn import replace_linear_with_wal
    
    print(f"  Loading model...")
    from .safe_load import load_torch

    model = load_torch(input_path, trust_pickle=trust_pickle)
    
    print(f"  Replacing nn.Linear with WAL (K={K}, C={C}, cached={cached})...")
    model = replace_linear_with_wal(model, K=K, C=C, cached=cached)
    
    print(f"  Saving to {output_path}...")
    torch.save(model, output_path)
    print(f"  Done.")
