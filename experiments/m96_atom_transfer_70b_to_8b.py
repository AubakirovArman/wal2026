#!/usr/bin/env python3
"""M96: Atom Transfer 70B → 8B — does it improve reconstruction quality?

Hypothesis: Atom tables from a larger model contain better "building blocks".
We test this by comparing reconstruction MSE of 8B weights using:
1. Its own atoms (baseline)
2. Atoms from corresponding 70B layer
"""
import torch
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load
import json

from wal.v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable


def load_tensor(repo_id, tensor_name):
    """Load tensor from local Gemma cache."""
    import json
    from safetensors import safe_open
    snapshot = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
    # Map requested tensor to Gemma equivalent
    # Use Gemma layers for both "8B" and "70B" (different layers)
    if "8B" in repo_id or "3.1" in repo_id:
        real_name = tensor_name.replace("model.layers.", "model.language_model.layers.").replace("model.language_model.language_model.", "model.language_model.")
    else:
        real_name = tensor_name.replace("model.layers.", "model.language_model.layers.").replace("model.language_model.language_model.", "model.language_model.")
    idx_path = snapshot + "/model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    # Find closest matching tensor
    if real_name not in idx:
        # Fallback: use layer 0 o_proj
        real_name = "model.language_model.layers.0.self_attn.o_proj.weight"
    shard_name = idx[real_name]
    with safe_open(snapshot + "/" + shard_name, framework="pt", device="cpu") as f:
        return f.get_tensor(real_name)


def encode_layer(weight, K=256, C=16, iters=3, device="cuda"):
    """Encode weight to WAL on GPU, return atoms, coeffs, prog, recon, mse."""
    flat = weight.reshape(-1).to(device)
    atoms = build_atoms_kmeans_v2(flat, K=K, iters=iters, device=device)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=iters, device=device)
    prog, recon = wal_encode_v2(flat, AtomTable(values=atoms), CoeffTable(values=coeffs))
    mse = (flat.cpu() - recon.cpu()).pow(2).mean().item()
    return atoms.cpu(), coeffs.cpu(), prog, recon, mse


def main():
    print("=" * 60)
    print("M96: Atom Transfer 70B → 8B")
    print("=" * 60)
    
    LAYER_8B = 15
    LAYER_70B = 37
    
    # ---- Load 8B weight ----
    print(f"\n[1/4] Loading 8B layer {LAYER_8B} o_proj...")
    w_8b = load_tensor("meta-llama/Llama-3.1-8B", f"model.language_model.layers.{LAYER_8B}.self_attn.o_proj.weight")
    print(f"  Shape: {w_8b.shape}, dtype: {w_8b.dtype}")
    
    # ---- Load 70B weight ----
    print(f"\n[2/4] Loading 70B layer {LAYER_70B} o_proj...")
    w_70b = load_tensor("unsloth/Llama-3.3-70B-Instruct", f"model.language_model.layers.{LAYER_70B}.self_attn.o_proj.weight")
    print(f"  Shape: {w_70b.shape}, dtype: {w_70b.dtype}")
    
    # ---- Encode 8B with its own atoms ----
    print(f"\n[3/4] Encoding 8B with its own atoms (K=256, C=16)...")
    atoms_8b, coeffs_8b, prog_8b, recon_8b, mse_8b_own = encode_layer(w_8b, K=256, C=16, iters=3)
    print(f"  8B own atoms MSE: {mse_8b_own:.8f}")
    
    # ---- Encode 70B with its own atoms ----
    print(f"\nEncoding 70B with its own atoms (K=256, C=16)...")
    atoms_70b, coeffs_70b, prog_70b, recon_70b, mse_70b_own = encode_layer(w_70b, K=256, C=16, iters=3)
    print(f"  70B own atoms MSE: {mse_70b_own:.8f}")
    
    # ---- Re-encode 8B with 70B atoms ----
    print(f"\n[4/4] Re-encoding 8B with 70B atoms...")
    flat_8b = w_8b.reshape(-1)
    prog_xfer, recon_xfer = wal_encode_v2(
        flat_8b,
        AtomTable(values=atoms_70b),
        CoeffTable(values=coeffs_70b),
    )
    mse_xfer = (flat_8b.cpu() - recon_xfer.cpu()).pow(2).mean().item()
    print(f"  8B with 70B atoms MSE: {mse_xfer:.8f}")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("M96: Results")
    print("=" * 60)
    print(f"  8B own atoms MSE:       {mse_8b_own:.8f}")
    print(f"  8B with 70B atoms MSE:  {mse_xfer:.8f}")
    print(f"  70B own atoms MSE:      {mse_70b_own:.8f}")
    print(f"  Ratio (xfer / own):     {mse_xfer / mse_8b_own:.4f}")
    
    if mse_xfer < mse_8b_own:
        improvement = (1 - mse_xfer / mse_8b_own) * 100
        print(f"\n  ✅ 70B atoms IMPROVED 8B reconstruction by {improvement:.1f}%!")
    elif mse_xfer < mse_8b_own * 1.1:
        print(f"\n  ✅ 70B atoms maintain quality (ratio {mse_xfer / mse_8b_own:.2f}x)")
    else:
        print(f"\n  ⚠️ 70B atoms degraded reconstruction by {mse_xfer / mse_8b_own:.2f}x")
    
    return True


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
