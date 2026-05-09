#!/usr/bin/env python3
"""WAL Framework — Phase 1: Encoding."""
import torch
from pathlib import Path


def encode_model(
    input_path: str,
    output_dir: str,
    K: int = 256,
    C: int = 16,
    device: str = "cuda",
    trust_pickle: bool = False,
):
    """Encode a PyTorch model to WAL format.
    
    Args:
        input_path: Path to model checkpoint (.pt or .safetensors)
        output_dir: Output directory for WAL-encoded model
        K: Number of atoms
        C: Number of coefficients
        device: Device for encoding
    """
    from wal.v1.nn import replace_linear_with_wal, wal_state_dict
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"  Loading model from {input_path}...")
    from .safe_load import load_torch

    model = load_torch(input_path, trust_pickle=trust_pickle)
    if isinstance(model, dict):
        # state_dict
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained("unsloth/Llama-3.3-70B-Instruct")
        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(model)
    
    # Replace layers
    print(f"  Encoding with K={K}, C={C}...")
    model = replace_linear_with_wal(model, K=K, C=C, cached=True)
    
    # Save state dict
    print(f"  Saving WAL state dict...")
    state = wal_state_dict(model)
    torch.save(state, output_dir / "wal_state.pt")
    
    print(f"  Done. Output: {output_dir}")


def encode_tensor(tensor: torch.Tensor, K: int = 256, C: int = 16, device: str = "cuda"):
    """Encode a single tensor to WAL format.
    
    Returns:
        (prog, atom_table, coeffs) — WAL representation
    """
    from wal.v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
    from wal.v2.isa import AtomTable, CoeffTable
    
    target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    flat = tensor.to(target).reshape(-1)
    atoms = AtomTable(build_atoms_kmeans_v2(flat, K=K, iters=5, device=target))
    coeffs = CoeffTable(build_coeff_table(flat, atoms.values, C=C, iters=5, device=target))
    prog, recon = wal_encode_v2(flat, atoms, coeffs, shape=tensor.shape)
    return prog, atoms, coeffs, recon
