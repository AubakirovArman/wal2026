#!/usr/bin/env python3
"""M59a: WAL-0 global codebook across all layers.

Encode all parameters, collect all programs into a single global codebook.
Measure: global vocabulary size, entropy, coverage.
Compare to per-layer codebooks (sum of per-layer unique programs).
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000
SPIKY_THRESHOLD = 0.08

print("Loading model...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

def pack_programs(indices, signs):
    sign_map = signs.clone()
    sign_map[signs == -1] = 0
    sign_map[signs == 0] = 1
    sign_map[signs == 1] = 2
    packed = (sign_map[:, 0].long() << 16) | (indices[:, 0].long() << 9) | \
             (sign_map[:, 1].long() << 7) | (indices[:, 1].long())
    return packed

print("\nEncoding all layers and collecting programs...")
stats = {"encoded": 0, "skipped": 0}
all_programs = []
per_layer_unique = []
t0_total = time.time()

for idx, (name, param) in enumerate(list(model.named_parameters())):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue
    
    w = param.data.float()
    std = (w / w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)).std().item()
    is_spiky = std < SPIKY_THRESHOLD
    
    if not is_spiky:
        param_device = param.device
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        if flat.numel() > SAMPLE_SIZE:
            idx_samp = torch.randperm(flat.numel(), device=param_device)[:SAMPLE_SIZE]
            samples = flat[idx_samp]
        else:
            samples = flat
        
        atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=param_device)
        atoms = atoms.to(param_device)
        
        indices, signs, recon = wal_encode_scalar_gpu(flat.to(param_device), atoms, L_MAX)
        packed = pack_programs(indices, signs)
        
        all_programs.append(packed.cpu())
        per_layer_unique.append(torch.unique(packed).numel())
        
        # Clean up GPU memory
        del indices, signs, recon, packed, atoms, samples, flat, w_norm, row_scale, w
        torch.cuda.empty_cache()
        
        stats["encoded"] += 1
        if stats["encoded"] % 50 == 0:
            print(f"  Encoded {stats['encoded']} params... elapsed={time.time()-t0_total:.0f}s")
    else:
        stats["skipped"] += 1

print(f"\nEncoding done in {time.time() - t0_total:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")

# Combine all programs
t0 = time.time()
all_packed = torch.cat(all_programs)
total_weights = all_packed.numel()
print(f"\nTotal weights across all layers: {total_weights:,}")

# Global codebook
global_unique, global_counts = torch.unique(all_packed, return_counts=True)
global_num_unique = global_unique.numel()

print(f"\n=== Global Codebook Analysis ===")
print(f"Per-layer unique programs (sum): {sum(per_layer_unique):,}")
print(f"Per-layer avg: {sum(per_layer_unique)/len(per_layer_unique):.0f}")
print(f"Per-layer median: {sorted(per_layer_unique)[len(per_layer_unique)//2]}")
print(f"Per-layer min: {min(per_layer_unique)}")
print(f"Per-layer max: {max(per_layer_unique)}")
print(f"")
print(f"Global unique programs: {global_num_unique:,}")
print(f"Global vs per-layer sum: {global_num_unique / sum(per_layer_unique) * 100:.1f}%")
print(f"Sharing ratio: {1 - global_num_unique / sum(per_layer_unique):.1%}")

# Entropy
global_probs = global_counts.float() / global_counts.sum()
H_global = -(global_probs * torch.log2(global_probs.clamp_min(1e-10))).sum().item()
max_H = torch.log2(torch.tensor(float(global_num_unique))).item()
print(f"\nGlobal entropy: {H_global:.2f} bits (max: {max_H:.2f})")
print(f"Effective bits/weight: {H_global:.2f}")

# Top-K coverage
g_sorted_counts, g_sorted_idx = torch.sort(global_counts, descending=True)
g_cumsum = g_sorted_counts.cumsum(dim=0)
for k in [1, 10, 100, 1000, 10000, 50000, 100000]:
    if k <= global_num_unique:
        cov = g_cumsum[k-1].item() / total_weights
        print(f"  Top-{k:6d} coverage: {cov:.4%}")

# Per-layer overlap analysis
print(f"\n=== Cross-layer sharing analysis ===")
# Sample 10 random layers and measure Jaccard similarity
import random
sample_layers = random.sample(range(len(all_programs)), min(10, len(all_programs)))
for i in range(len(sample_layers)):
    for j in range(i+1, len(sample_layers)):
        li, lj = sample_layers[i], sample_layers[j]
        set_i = all_programs[li]
        set_j = all_programs[lj]
        # Jaccard = |intersection| / |union|
        # On GPU: use torch.unique on combined
        combined = torch.cat([set_i, set_j])
        unique_combined = torch.unique(combined)
        intersection_size = len(set_i) + len(set_j) - len(unique_combined)
        union_size = len(unique_combined)
        jaccard = intersection_size / union_size
        print(f"  Layer {li} vs {lj}: Jaccard = {jaccard:.4f}")

# Estimate global compression
print(f"\n=== Compression Estimate ===")
print(f"Raw WAL-0: 16 bits/weight")
print(f"Per-layer codebook (avg): {sum(per_layer_unique)/len(per_layer_unique) * L_MAX * 2 / 1024:.1f} KB table per layer")
print(f"Global codebook ID: {max_H:.2f} bits/weight")
print(f"Global codebook table: {global_num_unique * L_MAX * 2 / 1024:.1f} KB")
print(f"Global precomputed recon table: {global_num_unique * 4 / 1024:.1f} KB")

print(f"\nM59a complete. Analysis time: {time.time()-t0:.1f}s")
