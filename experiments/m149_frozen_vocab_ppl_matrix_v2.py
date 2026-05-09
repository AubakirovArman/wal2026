"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M149 v2 — Frozen Vocabulary PPL Matrix (fast version).

Radically simplified: CPU model, 2 layers, K=64, iters=1.
"""
import torch
import time
import json
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def encode_layer(w, atoms, coeffs):
    flat = w.reshape(-1)
    t0 = time.time()
    prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
    return {
        'mse': ((flat - recon) ** 2).mean().item(),
        'encode_time': time.time() - t0,
        'prog': prog,
    }


def main():
    print("=" * 60)
    print("M149 v2 — Frozen Vocab PPL Matrix (fast)")
    print("=" * 60)
    
    print("\nLoading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    layers = [0, 16]
    modules = ['q_proj', 'v_proj']
    K, C = 64, 8
    
    # Build global table from 2 layers
    print("\nBuilding global atom table...")
    weights = []
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data
            weights.append(w.reshape(-1).float())
    
    all_flat = torch.cat(weights)
    t0 = time.time()
    global_atoms = build_l0_atoms(all_flat, K=K, iters=1)
    global_coeffs = build_coeff_table(all_flat, global_atoms, C=C, iters=1)
    global_atoms = global_atoms[torch.argsort(global_atoms.abs())]
    print(f"Global table build: {time.time()-t0:.1f}s")
    
    results = []
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            path = f"layers.{li}.self_attn.{m}"
            
            # Rebuilt
            atoms_r = build_l0_atoms(w.reshape(-1), K=K, iters=1)
            coeffs_r = build_coeff_table(w.reshape(-1), atoms_r, C=C, iters=1)
            atoms_r = atoms_r[torch.argsort(atoms_r.abs())]
            rebuilt = encode_layer(w, atoms_r, coeffs_r)
            
            # Frozen
            frozen = encode_layer(w, global_atoms, global_coeffs)
            
            # Synthetic edit
            w_edit = w + torch.randn_like(w) * 0.001
            rebuilt_e = encode_layer(w_edit, atoms_r, coeffs_r)
            frozen_e = encode_layer(w_edit, global_atoms, global_coeffs)
            
            diff_r = (rebuilt['prog'].atom_ids != rebuilt_e['prog'].atom_ids).float().mean().item()
            diff_f = (frozen['prog'].atom_ids != frozen_e['prog'].atom_ids).float().mean().item()
            
            print(f"  {path}: rebuilt_mse={rebuilt['mse']:.2e}, frozen_mse={frozen['mse']:.2e}, "
                  f"ratio={frozen['mse']/max(rebuilt['mse'],1e-10):.2f}, diff_r={diff_r:.3f}, diff_f={diff_f:.3f}")
            
            results.append({
                'path': path,
                'rebuilt_mse': rebuilt['mse'],
                'frozen_mse': frozen['mse'],
                'ratio': frozen['mse'] / max(rebuilt['mse'], 1e-10),
                'rebuilt_diff': diff_r,
                'frozen_diff': diff_f,
            })
    
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    avg_rdiff = sum(r['rebuilt_diff'] for r in results) / len(results)
    avg_fdiff = sum(r['frozen_diff'] for r in results) / len(results)
    
    print(f"\nAvg frozen/rebuilt ratio: {avg_ratio:.3f}")
    print(f"Avg diff (rebuilt): {avg_rdiff:.3f}")
    print(f"Avg diff (frozen):  {avg_fdiff:.3f}")
    
    out = {'avg_ratio': avg_ratio, 'avg_rebuilt_diff': avg_rdiff, 'avg_frozen_diff': avg_fdiff, 'layers': results}
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m149_frozen_vocab_ppl_matrix.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
