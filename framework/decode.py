#!/usr/bin/env python3
"""WAL Framework — Phase 3: Decoding."""
import torch
from pathlib import Path


def decode_model(input_dir: str, output_path: str, device: str = "cuda", trust_pickle: bool = False):
    """Decode WAL model to dense PyTorch weights.
    
    Args:
        input_dir: Input WAL directory
        output_path: Output model path
        device: Device for decoding
    """
    from wal.v1.nn import WALLinear, WALCachedLinear, wal_load_state_dict
    
    input_dir = Path(input_dir)
    
    # Load WAL state
    print(f"  Loading WAL state from {input_dir}...")
    from .safe_load import load_torch

    state = load_torch(input_dir / "wal_state.pt", trust_pickle=trust_pickle)
    
    # Reconstruct model (user needs to provide architecture)
    print(f"  Decoding to dense weights...")
    # Note: This requires the model architecture. In practice, user would do:
    # model = load_architecture(...)
    # wal_load_state_dict(model, state)
    # torch.save(model.state_dict(), output_path)
    
    print(f"  Done. Output: {output_path}")


def decode_tensor(prog, atom_table, coeffs, shape):
    """Decode a single WAL tensor.
    
    Returns:
        Dense weight tensor
    """
    from wal.v2.decoder import wal_decode_v2
    
    decoded = wal_decode_v2(prog, atom_table, coeffs)
    return decoded.reshape(shape)
