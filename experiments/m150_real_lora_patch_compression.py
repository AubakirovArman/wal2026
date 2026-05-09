#!/usr/bin/env python3
"""M150 — Real LoRA Patch Compression.

Trains real LoRA edits (rank=4,8 × steps=50,100) then measures WAL patch sizes.
Compares against synthetic random perturbation (M139 baseline).
"""
import torch
import torch.nn as nn
import time
import json
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def train_lora_edit(model, tokenizer, target_modules, rank=4, steps=50, lr=1e-4, device='cuda'):
    """Train a simple LoRA edit on a small dataset."""
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Inject LoRA
    lora_params = {}
    for name, module in model.named_modules():
        if any(m in name for m in target_modules):
            if isinstance(module, nn.Linear):
                in_d = module.in_features
                out_d = module.out_features
                A = nn.Parameter(torch.randn(in_d, rank, device=device) * 0.01)
                B = nn.Parameter(torch.zeros(rank, out_d, device=device))
                lora_params[name] = (A, B)
                module.register_parameter('lora_A', A)
                module.register_parameter('lora_B', B)
    
    # Simple synthetic target: shift outputs toward a direction
    optimizer = torch.optim.Adam([p for pair in lora_params.values() for p in pair], lr=lr)
    
    model.train()
    for step in range(steps):
        # Synthetic batch: random inputs, target = shifted output
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward with LoRA
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Extract deltas
    deltas = {}
    for name, (A, B) in lora_params.items():
        deltas[name] = (A @ B).detach().clone()
    
    return deltas


def build_wal_patch(base_weights, edited_weights, global_atoms, global_coeffs):
    """Build WAL patch: bitmask + changed atom/coeff IDs."""
    patches = {}
    total_bytes = 0
    
    for name in base_weights:
        w_base = base_weights[name].reshape(-1)
        w_edit = edited_weights[name].reshape(-1)
        
        prog_base, _ = wal_encode_v1(w_base, global_atoms, global_coeffs, batch=262_144)
        prog_edit, _ = wal_encode_v1(w_edit, global_atoms, global_coeffs, batch=262_144)
        
        changed = (prog_base.atom_ids != prog_edit.atom_ids) | (prog_base.coeff_ids != prog_edit.coeff_ids)
        n_changed = changed.sum().item()
        n_total = changed.numel()
        
        # Bitmask patch
        bitmask_bytes = (n_total + 7) // 8
        new_atom_bytes = n_changed * 1
        new_coeff_bytes = n_changed * 1
        patch_bytes = bitmask_bytes + new_atom_bytes + new_coeff_bytes
        
        patches[name] = {
            'n_total': n_total,
            'n_changed': n_changed,
            'change_ratio': n_changed / n_total,
            'patch_bytes': patch_bytes,
        }
        total_bytes += patch_bytes
    
    return patches, total_bytes


def main():
    print("=" * 60)
    print("M150 — Real LoRA Patch Compression")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("\nLoading Llama-3.1-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Target modules
    target_modules = ['q_proj', 'v_proj']
    
    # Build global atom table from target layers
    print("\nBuilding global atom table...")
    layer_indices = list(range(0, 32, 4))  # Every 4th layer
    all_weights = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for mod_name in target_modules:
            w = getattr(layer.self_attn, mod_name).weight.data
            all_weights.append(w.reshape(-1).cpu().float())
    
    all_flat = torch.cat(all_weights)
    global_atoms = build_l0_atoms(all_flat, K=256, iters=2)
    global_coeffs = build_coeff_table(all_flat, global_atoms, C=16, iters=2)
    global_atoms = global_atoms[torch.argsort(global_atoms.abs())]
    
    # Extract base weights
    base_weights = {}
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for mod_name in target_modules:
            name = f"layers.{layer_idx}.self_attn.{mod_name}"
            base_weights[name] = getattr(layer.self_attn, mod_name).weight.data.clone()
    
    configs = [
        {'rank': 4, 'steps': 50},
        {'rank': 4, 'steps': 100},
        {'rank': 8, 'steps': 50},
        {'rank': 8, 'steps': 100},
    ]
    
    results = []
    
    for cfg in configs:
        rank, steps = cfg['rank'], cfg['steps']
        print(f"\n--- Training LoRA rank={rank}, steps={steps} ---")
        
        t0 = time.time()
        deltas = train_lora_edit(model, tokenizer, target_modules, rank=rank, steps=steps, device=device)
        train_time = time.time() - t0
        
        # Apply deltas
        edited_weights = {}
        for name, delta in deltas.items():
            edited_weights[name] = base_weights[name] + delta
        
        # Build patch
        t0 = time.time()
        patches, total_bytes = build_wal_patch(base_weights, edited_weights, global_atoms, global_coeffs)
        patch_time = time.time() - t0
        
        # LoRA size
        lora_params = 0
        for name, delta in deltas.items():
            layer_name = name.rsplit('.', 1)[0]
            in_d = delta.shape[0] if len(delta.shape) == 2 else 1
            out_d = delta.shape[1] if len(delta.shape) == 2 else 1
            lora_params += in_d * rank + rank * out_d
        lora_bytes = lora_params * 2  # bf16
        
        avg_change = sum(p['change_ratio'] for p in patches.values()) / len(patches)
        
        result = {
            'rank': rank,
            'steps': steps,
            'train_time': train_time,
            'patch_time': patch_time,
            'wal_patch_bytes': total_bytes,
            'wal_patch_mb': total_bytes / 1e6,
            'lora_bytes': lora_bytes,
            'lora_mb': lora_bytes / 1e6,
            'ratio_wal_to_lora': total_bytes / max(lora_bytes, 1),
            'avg_change_ratio': avg_change,
            'patches': patches,
        }
        results.append(result)
        
        print(f"  WAL patch: {total_bytes/1e6:.2f} MB")
        print(f"  LoRA size: {lora_bytes/1e6:.2f} MB")
        print(f"  WAL/LoRA ratio: {total_bytes/max(lora_bytes,1):.1f}×")
        print(f"  Avg change ratio: {avg_change:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Rank':>6} {'Steps':>6} {'WAL MB':>10} {'LoRA MB':>10} {'Ratio':>8} {'Change':>8}")
    for r in results:
        print(f"{r['rank']:>6} {r['steps']:>6} {r['wal_patch_mb']:>10.2f} {r['lora_mb']:>10.2f} {r['ratio_wal_to_lora']:>8.1f} {r['avg_change_ratio']:>8.3f}")
    
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m150_real_lora_patch_compression.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
