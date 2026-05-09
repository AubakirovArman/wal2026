"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M150 v2 — Real LoRA Patch Compression (synthetic deltas, CPU).

Generates structured low-rank deltas, applies to weights, measures WAL patch size.
"""
import torch
import json
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def main():
    print("=" * 60)
    print("M150 v2 — LoRA Patch Compression")
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
    
    # Build global atom table
    print("\nBuilding global atom table...")
    weights = []
    for li in layers:
        for m in modules:
            w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
            weights.append(w)
    
    all_flat = torch.cat([w.reshape(-1) for w in weights])
    atoms = build_l0_atoms(all_flat, K=K, iters=1)
    coeffs = build_coeff_table(all_flat, atoms, C=C, iters=1)
    atoms = atoms[torch.argsort(atoms.abs())]
    
    configs = [
        {'rank': 1, 'scale': 0.01},
        {'rank': 4, 'scale': 0.01},
        {'rank': 8, 'scale': 0.01},
        {'rank': 4, 'scale': 0.05},
        {'rank': 8, 'scale': 0.05},
    ]
    
    results = []
    
    for cfg in configs:
        rank, scale = cfg['rank'], cfg['scale']
        print(f"\n--- rank={rank}, scale={scale} ---")
        
        total_patch = 0
        total_lora = 0
        total_changed = 0
        total_weights = 0
        
        for li in layers:
            for m in modules:
                w = getattr(model.model.layers[li].self_attn, m).weight.data.float()
                out_d, in_d = w.shape
                
                # Generate LoRA delta
                A = torch.randn(out_d, rank) * 0.01
                B = torch.randn(rank, in_d) * 0.01
                delta = A @ B * scale
                w_edit = w + delta
                
                # Encode both
                flat = w.reshape(-1)
                flat_e = w_edit.reshape(-1)
                prog, _ = wal_encode_v1(flat, atoms, coeffs, batch=65_536)
                prog_e, _ = wal_encode_v1(flat_e, atoms, coeffs, batch=65_536)
                
                # Patch stats
                changed = ((prog.atom_ids != prog_e.atom_ids) | (prog.coeff_ids != prog_e.coeff_ids))
                n_changed = changed.sum().item()
                n_total = changed.numel()
                
                # Bitmask patch size
                bitmask_bytes = (n_total + 7) // 8 + n_changed * 2
                
                # LoRA size (bf16)
                lora_bytes = (out_d * rank + rank * in_d) * 2
                
                total_patch += bitmask_bytes
                total_lora += lora_bytes
                total_changed += n_changed
                total_weights += n_total
                
                print(f"  layers.{li}.{m}: changed={n_changed/n_total:.3f}, patch={bitmask_bytes/1e6:.3f}MB, lora={lora_bytes/1e3:.1f}KB")
        
        ratio = total_patch / max(total_lora, 1)
        print(f"  TOTAL: patch={total_patch/1e6:.3f}MB, lora={total_lora/1e6:.3f}MB, ratio={ratio:.1f}x")
        
        results.append({
            'rank': rank,
            'scale': scale,
            'patch_mb': total_patch / 1e6,
            'lora_mb': total_lora / 1e6,
            'ratio': ratio,
            'avg_change': total_changed / max(total_weights, 1),
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Rank':>6} {'Scale':>8} {'Patch MB':>10} {'LoRA MB':>10} {'Ratio':>8} {'Change':>8}")
    for r in results:
        print(f"{r['rank']:>6} {r['scale']:>8.3f} {r['patch_mb']:>10.3f} {r['lora_mb']:>10.3f} {r['ratio']:>8.1f} {r['avg_change']:>8.3f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m150_real_lora_patch_compression.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
