#!/usr/bin/env python3
"""M149 — Frozen Vocabulary PPL Matrix.

Compares Raw-WAL rebuilt table vs frozen table across key metrics:
- MSE reconstruction per layer
- Program entropy
- Encode time
- Patch locality (synthetic edit diff)
"""
import torch
import time
import json
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1, wal_decode_v1


def compute_entropy(ids: torch.Tensor, vocab_size: int) -> float:
    """Compute normalized entropy of ID distribution."""
    counts = torch.bincount(ids.long(), minlength=vocab_size).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log2(probs)).sum().item()
    max_entropy = torch.log2(torch.tensor(float(vocab_size))).item()
    return entropy / max_entropy


def encode_layer_rebuilt(weight: torch.Tensor, K: int = 256, C: int = 16):
    """Encode with per-layer rebuilt atoms (CPU to avoid GPU OOM)."""
    flat = weight.reshape(-1).cpu().float()
    atoms = build_l0_atoms(flat, K=K, iters=2)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=2)
    atoms = atoms[torch.argsort(atoms.abs())]
    
    t0 = time.time()
    prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    encode_time = time.time() - t0
    
    mse = ((flat - recon) ** 2).mean().item()
    atom_entropy = compute_entropy(prog.atom_ids, K)
    coeff_entropy = compute_entropy(prog.coeff_ids, C)
    
    return {
        'mse': mse,
        'atom_entropy': atom_entropy,
        'coeff_entropy': coeff_entropy,
        'encode_time': encode_time,
        'atoms': atoms,
        'coeffs': coeffs,
        'prog': prog,
    }


def encode_layer_frozen(weight: torch.Tensor, global_atoms: torch.Tensor, global_coeffs: torch.Tensor):
    """Encode with frozen global atoms (CPU to avoid GPU OOM)."""
    flat = weight.reshape(-1).cpu().float()
    
    t0 = time.time()
    prog, recon = wal_encode_v1(flat, global_atoms.cpu(), global_coeffs.cpu(), batch=65_536)
    encode_time = time.time() - t0
    
    mse = ((flat - recon) ** 2).mean().item()
    K = global_atoms.numel()
    C = global_coeffs.numel()
    atom_entropy = compute_entropy(prog.atom_ids, K)
    coeff_entropy = compute_entropy(prog.coeff_ids, C)
    
    return {
        'mse': mse,
        'atom_entropy': atom_entropy,
        'coeff_entropy': coeff_entropy,
        'encode_time': encode_time,
        'prog': prog,
    }


def measure_patch_locality(base_prog, edit_prog):
    """Measure diff % between base and edited programs."""
    diff = (base_prog.atom_ids != edit_prog.atom_ids).float().mean().item()
    return diff


def main():
    print("=" * 60)
    print("M149 — Frozen Vocabulary PPL Matrix")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading Llama-3.1-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # Select representative layers and modules
    layers_to_test = [0, 16, 31]
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    K, C = 256, 16
    
    # --- Phase 1: Build global atom table ---
    print("\n--- Building global atom table ---")
    all_weights = []
    for layer_idx in layers_to_test:
        layer = model.model.layers[layer_idx]
        for mod_name in modules:
            if hasattr(layer.self_attn, mod_name):
                w = getattr(layer.self_attn, mod_name).weight.data
            elif hasattr(layer.mlp, mod_name):
                w = getattr(layer.mlp, mod_name).weight.data
            else:
                continue
            all_weights.append(w.reshape(-1).cpu().float())
    
    all_flat = torch.cat(all_weights).cpu()
    print(f"Total weights for global table: {all_flat.numel() / 1e9:.2f}B")
    
    t0 = time.time()
    global_atoms = build_l0_atoms(all_flat, K=K, iters=2)
    global_coeffs = build_coeff_table(all_flat, global_atoms, C=C, iters=2)
    global_atoms = global_atoms[torch.argsort(global_atoms.abs())]
    global_build_time = time.time() - t0
    print(f"Global table build time: {global_build_time:.1f}s")
    
    # --- Phase 2: Encode each layer both ways ---
    print("\n--- Encoding layers (rebuilt vs frozen) ---")
    results = []
    
    for layer_idx in layers_to_test:
        layer = model.model.layers[layer_idx]
        for mod_name in modules:
            if hasattr(layer.self_attn, mod_name):
                w = getattr(layer.self_attn, mod_name).weight.data
                path = f"layers.{layer_idx}.self_attn.{mod_name}"
            elif hasattr(layer.mlp, mod_name):
                w = getattr(layer.mlp, mod_name).weight.data
                path = f"layers.{layer_idx}.mlp.{mod_name}"
            else:
                continue
            
            # Rebuilt
            rebuilt = encode_layer_rebuilt(w, K=K, C=C)
            
            # Frozen
            frozen = encode_layer_frozen(w, global_atoms, global_coeffs)
            
            # Synthetic edit for patch locality
            w_edit = w.clone() + torch.randn_like(w) * 0.001
            rebuilt_edit = encode_layer_rebuilt(w_edit, K=K, C=C)
            frozen_edit = encode_layer_frozen(w_edit, global_atoms, global_coeffs)
            
            rebuilt_diff = measure_patch_locality(rebuilt['prog'], rebuilt_edit['prog'])
            frozen_diff = measure_patch_locality(frozen['prog'], frozen_edit['prog'])
            
            results.append({
                'path': path,
                'rebuilt_mse': rebuilt['mse'],
                'frozen_mse': frozen['mse'],
                'mse_delta': frozen['mse'] - rebuilt['mse'],
                'mse_ratio': frozen['mse'] / max(rebuilt['mse'], 1e-10),
                'rebuilt_atom_entropy': rebuilt['atom_entropy'],
                'frozen_atom_entropy': frozen['atom_entropy'],
                'rebuilt_coeff_entropy': rebuilt['coeff_entropy'],
                'frozen_coeff_entropy': frozen['coeff_entropy'],
                'rebuilt_encode_time': rebuilt['encode_time'],
                'frozen_encode_time': frozen['encode_time'],
                'rebuilt_patch_diff': rebuilt_diff,
                'frozen_patch_diff': frozen_diff,
            })
            
            print(f"  {path}: rebuilt_mse={rebuilt['mse']:.2e}, frozen_mse={frozen['mse']:.2e}, "
                  f"ratio={frozen['mse']/max(rebuilt['mse'],1e-10):.3f}, "
                  f"rebuilt_diff={rebuilt_diff:.3f}, frozen_diff={frozen_diff:.3f}")
    
    # --- Phase 3: Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_rebuilt_mse = sum(r['rebuilt_mse'] for r in results) / len(results)
    avg_frozen_mse = sum(r['frozen_mse'] for r in results) / len(results)
    avg_ratio = sum(r['mse_ratio'] for r in results) / len(results)
    avg_rebuilt_diff = sum(r['rebuilt_patch_diff'] for r in results) / len(results)
    avg_frozen_diff = sum(r['frozen_patch_diff'] for r in results) / len(results)
    
    print(f"\nAverage MSE:")
    print(f"  Rebuilt (per-layer):  {avg_rebuilt_mse:.2e}")
    print(f"  Frozen (global):      {avg_frozen_mse:.2e}")
    print(f"  Ratio frozen/rebuilt: {avg_ratio:.3f}")
    
    print(f"\nAverage Patch Diff (synthetic edit):")
    print(f"  Rebuilt table:        {avg_rebuilt_diff:.3f}")
    print(f"  Frozen table:         {avg_frozen_diff:.3f}")
    
    print(f"\nEntropy:")
    avg_rebuilt_aent = sum(r['rebuilt_atom_entropy'] for r in results) / len(results)
    avg_frozen_aent = sum(r['frozen_atom_entropy'] for r in results) / len(results)
    print(f"  Atom entropy rebuilt: {avg_rebuilt_aent:.4f}")
    print(f"  Atom entropy frozen:  {avg_frozen_aent:.4f}")
    
    # Save results
    summary = {
        'K': K, 'C': C,
        'global_build_time': global_build_time,
        'n_layers_tested': len(layers_to_test),
        'avg_rebuilt_mse': avg_rebuilt_mse,
        'avg_frozen_mse': avg_frozen_mse,
        'avg_ratio': avg_ratio,
        'avg_rebuilt_patch_diff': avg_rebuilt_diff,
        'avg_frozen_patch_diff': avg_frozen_diff,
        'avg_rebuilt_atom_entropy': avg_rebuilt_aent,
        'avg_frozen_atom_entropy': avg_frozen_aent,
        'per_layer': results,
    }
    
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m149_frozen_vocab_ppl_matrix.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
