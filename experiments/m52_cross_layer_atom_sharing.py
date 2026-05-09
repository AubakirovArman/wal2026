#!/usr/bin/env python3
"""M52: Cross-layer atom sharing — analyze atom similarity and shared quality."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal import build_atoms_kmeans, wal_encode_scalar


def test():
    device = torch.device('cuda:2')
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    layer_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    param_name_tpl = "model.layers.{}.self_attn.o_proj.weight"
    K = 128
    lmax = 2
    
    print("=" * 60)
    print("M52: Cross-Layer Atom Sharing")
    print("=" * 60)
    
    print(f"\n[1] Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    
    # Collect per-layer atoms and weights
    print(f"\n[2] Building per-layer atoms for {len(layer_indices)} layers...")
    per_layer_data = {}
    
    for idx in layer_indices:
        name = param_name_tpl.format(idx)
        param = dict(model.named_parameters())[name]
        w = param.data.float().to(device)
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        SAMPLE_SIZE = 1_000_000
        if flat.numel() > SAMPLE_SIZE:
            idx_samp = torch.randperm(flat.numel())[:SAMPLE_SIZE]
            samples = flat[idx_samp]
        else:
            samples = flat
        
        atoms = build_atoms_kmeans(samples, K, iters=5, device=device)
        per_layer_data[idx] = {
            'w': w, 'w_norm': w_norm, 'row_scale': row_scale,
            'atoms': atoms.to(device),
        }
        print(f"    Layer {idx}: atoms built")
    
    # Compute atom similarity matrix
    print(f"\n[3] Atom similarity (cosine) between layers...")
    n = len(layer_indices)
    sim_matrix = torch.zeros(n, n)
    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            a = per_layer_data[li]['atoms']
            b = per_layer_data[lj]['atoms']
            # Pairwise max cosine (optimal matching)
            a_norm = a / a.abs().amax()
            b_norm = b / b.abs().amax()
            dots = (a_norm.unsqueeze(1) * b_norm.unsqueeze(0)).sum(dim=-1)  # [K, K]
            # Greedy matching
            matched = 0.0
            used = set()
            vals, ids = dots.flatten().sort(descending=True)
            for val, flat_idx in zip(vals, ids):
                ii = flat_idx.item() // K
                jj = flat_idx.item() % K
                if ii not in used and jj not in used:
                    matched += val.item()
                    used.add(ii)
                    used.add(jj)
                if len(used) >= 2 * K:
                    break
            sim_matrix[i, j] = matched / K
    
    print(f"    Similarity matrix:")
    for i, li in enumerate(layer_indices):
        row_str = " ".join([f"{sim_matrix[i, j]:.3f}" for j in range(n)])
        print(f"    L{li:2d}: {row_str}")
    
    # Build shared atoms from pooled samples
    print(f"\n[4] Building shared atoms (pooled from all layers)...")
    pooled = torch.cat([per_layer_data[idx]['w_norm'].reshape(-1) for idx in layer_indices])
    if pooled.numel() > 2_000_000:
        idx_samp = torch.randperm(pooled.numel())[:2_000_000]
        pooled_samples = pooled[idx_samp]
    else:
        pooled_samples = pooled
    
    shared_atoms = build_atoms_kmeans(pooled_samples, K, iters=5, device=device).to(device)
    print(f"    Shared atoms built from {pooled_samples.numel() / 1e6:.1f}M samples")
    
    # Compare per-layer vs shared encoding quality
    print(f"\n[5] Encoding quality: per-layer vs shared atoms...")
    print(f"    {'Layer':>6} {'Per-layer relMSE':>18} {'Shared relMSE':>18} {'Ratio':>8}")
    
    total_per_layer_bytes = 0
    total_shared_bytes = K * 4  # one shared atom table
    
    for idx in layer_indices:
        data = per_layer_data[idx]
        w = data['w']
        w_norm = data['w_norm']
        row_scale = data['row_scale']
        flat = w_norm.reshape(-1)
        
        # Per-layer encode
        _, recon_pl = wal_encode_scalar(flat, data['atoms'], lmax)
        recon_pl = recon_pl.reshape(w.shape) * row_scale
        relmse_pl = ((w - recon_pl) ** 2).mean() / (w ** 2).mean()
        
        # Shared encode
        _, recon_sh = wal_encode_scalar(flat, shared_atoms, lmax)
        recon_sh = recon_sh.reshape(w.shape) * row_scale
        relmse_sh = ((w - recon_sh) ** 2).mean() / (w ** 2).mean()
        
        ratio = relmse_sh.item() / relmse_pl.item()
        print(f"    {idx:>6} {relmse_pl.item():>18.8f} {relmse_sh.item():>18.8f} {ratio:>8.2f}")
        
        total_per_layer_bytes += K * 4  # per-layer atom table
        total_shared_bytes += flat.numel() * lmax * 2  # programs only
    
    print(f"\n[6] Storage comparison for {len(layer_indices)} layers...")
    original_bytes = sum(per_layer_data[idx]['w'].numel() * 2 for idx in layer_indices)
    print(f"    Original weights:       {original_bytes / 1e6:.1f} MB")
    print(f"    Per-layer atoms + prog: {(total_per_layer_bytes + original_bytes) / 1e6:.1f} MB")
    print(f"    Shared atoms + prog:    {total_shared_bytes / 1e6:.1f} MB")
    print(f"    Shared compression:     {original_bytes / total_shared_bytes:.2f}x")
    
    print("\n" + "=" * 60)
    print("M52: DONE")
    print("=" * 60)


if __name__ == "__main__":
    test()
