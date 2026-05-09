"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M59: Fast global codebook analysis using per-layer atoms (no global k-means bottleneck).

Encode is identical to M57 (per-layer atoms, PPL 2.7828 proven).
Only adds global codebook mining and compression analysis.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

model_name = "unsloth/Llama-3.3-70B-Instruct"
K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000
SPIKY_THRESHOLD = 0.08

print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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

print("\nEncoding all layers (per-layer atoms)...")
all_programs_cpu = []
stats = {"encoded": 0, "skipped": 0, "total_weights": 0}
t0 = time.time()

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
        
        all_programs_cpu.append(packed.cpu())
        stats["total_weights"] += packed.numel()
        
        # Apply recon to model (same as M57)
        w_hat = recon.reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        # Cleanup GPU memory
        del indices, signs, recon, packed, atoms, samples, flat, w_norm, row_scale, w
        if idx % 20 == 0:
            torch.cuda.empty_cache()
        
        stats["encoded"] += 1
        if stats["encoded"] % 50 == 0:
            print(f"  Encoded {stats['encoded']} params... elapsed={time.time()-t0:.0f}s")
    else:
        stats["skipped"] += 1

print(f"\nEncode done in {time.time()-t0:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Total weights: {stats['total_weights']:,}")

# ========================================================================
# Global codebook analysis
# ========================================================================
print("\n=== Global Codebook Analysis ===")
t0 = time.time()

# Use bincount instead of unique — much faster for small value range
all_packed = torch.cat(all_programs_cpu)
max_val = all_packed.max().item()
print(f"  Max packed value: {max_val}")

global_counts = torch.bincount(all_packed, minlength=max_val + 1)
nonzero_mask = global_counts > 0
global_counts = global_counts[nonzero_mask]
global_num_unique = global_counts.numel()

print(f"Global unique programs: {global_num_unique:,}")
print(f"Global vs total weights: {global_num_unique / stats['total_weights'] * 100:.4f}%")

# Entropy
global_probs = global_counts.float() / global_counts.sum()
H_global = -(global_probs * torch.log2(global_probs.clamp_min(1e-10))).sum().item()
max_H = torch.log2(torch.tensor(float(global_num_unique))).item()
print(f"Global entropy: {H_global:.2f} bits (max: {max_H:.2f})")

# Top-K coverage
g_sorted_counts, g_sorted_idx = torch.sort(global_counts, descending=True)
g_cumsum = g_sorted_counts.cumsum(dim=0)
for k in [1, 10, 100, 1000, 10000, 100000, 500000]:
    if k <= global_num_unique:
        cov = g_cumsum[k-1].item() / stats["total_weights"]
        print(f"  Top-{k:6d} coverage: {cov:.4%}")

# Compression estimate
original_size_gb = sum(p.numel() * 2 for p in model.parameters()) / 1e9
bits_per_id = max_H
program_bytes = stats["total_weights"] * bits_per_id / 8
codebook_bytes = global_num_unique * L_MAX * 2  # atom + sign per step
atom_table_bytes = K_ATOMS * 4

# Skipped weights (bf16)
skipped_weights = sum(p.numel() for n, p in model.named_parameters() 
                      if any((x[0]==n and len(x[1])==2 and "embed" not in n and "lm_head" not in n) for x in [(n, p.shape)]))
# Actually simpler: skipped weights = sum of skipped params
skipped_weights = 0
for name, param in model.named_parameters():
    if len(param.shape) != 2 or "embed_tokens" in name or "lm_head" in name:
        skipped_weights += param.numel()

skipped_bytes = skipped_weights * 2

# Row scales: approx 8192 rows per layer * 4 bytes * encoded layers
row_scale_bytes = stats["encoded"] * 8192 * 4

compressed_bytes = program_bytes + codebook_bytes + atom_table_bytes + skipped_bytes + row_scale_bytes

print(f"\n=== Compression Estimate ===")
print(f"Original size: {original_size_gb:.2f} GB")
print(f"Program IDs ({bits_per_id:.1f} bits): {program_bytes / 1e9:.2f} GB")
print(f"Codebook table: {codebook_bytes / 1024:.1f} KB")
print(f"Atom table: {atom_table_bytes / 1024:.1f} KB")
print(f"Skipped weights: {skipped_bytes / 1e9:.2f} GB")
print(f"Row scales: {row_scale_bytes / 1e9:.2f} GB")
print(f"Compressed total: {compressed_bytes / 1e9:.2f} GB")
print(f"Compression ratio: {original_size_gb / (compressed_bytes / 1e9):.2f}x")

print(f"\nM59 complete. Analysis time: {time.time()-t0:.1f}s")
