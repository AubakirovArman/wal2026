"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M59 v3: Fast global codebook analysis — ALL GPU, zero CPU materialization.

Per-layer encode + immediate GPU bincount. No 65B tensor ever touches CPU.
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
PACKED_MAX = 196480  # (2<<16)|(127<<9)|(2<<7)|127 + 1

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

print("\nEncoding all layers (per-layer atoms) + GPU bincount...")
layer_histograms = []  # each: CPU tensor of length PACKED_MAX
total_weights = 0
stats = {"encoded": 0, "skipped": 0}
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
        
        # GPU bincount — never materialize packed on CPU
        counts = torch.bincount(packed, minlength=PACKED_MAX)
        layer_histograms.append(counts.cpu())
        total_weights += packed.numel()
        
        # Apply recon to model
        w_hat = recon.reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        # Cleanup
        del indices, signs, recon, packed, counts, atoms, samples, flat, w_norm, row_scale, w
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
print(f"  Total weights: {total_weights:,}")

# ========================================================================
# Accumulate histograms on CPU (tiny tensors, negligible)
# ========================================================================
print("\n=== Global Codebook Analysis ===")
t1 = time.time()

global_counts = torch.zeros(PACKED_MAX, dtype=torch.int64)
for h in layer_histograms:
    global_counts += h

nonzero_mask = global_counts > 0
global_counts = global_counts[nonzero_mask]
global_num_unique = global_counts.numel()

print(f"Global unique programs: {global_num_unique:,}")
print(f"Global vs total weights: {global_num_unique / total_weights * 100:.4f}%")

# Entropy
probs = global_counts.float() / global_counts.sum()
H_global = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
max_H = torch.log2(torch.tensor(float(global_num_unique))).item()
print(f"Global entropy: {H_global:.2f} bits (max: {max_H:.2f})")

# Top-K coverage
g_sorted_counts, _ = torch.sort(global_counts, descending=True)
g_cumsum = g_sorted_counts.cumsum(dim=0)
for k in [1, 10, 100, 1000, 10000, 100000, 500000]:
    if k <= global_num_unique:
        cov = g_cumsum[k-1].item() / total_weights
        print(f"  Top-{k:6d} coverage: {cov:.4%}")

# Compression
original_size_gb = sum(p.numel() * 2 for p in model.parameters()) / 1e9
bits_per_id = max_H
program_bytes = total_weights * bits_per_id / 8
codebook_bytes = global_num_unique * L_MAX * 2
atom_table_bytes = K_ATOMS * 4

skipped_weights = 0
for name, param in model.named_parameters():
    if len(param.shape) != 2 or "embed_tokens" in name or "lm_head" in name:
        skipped_weights += param.numel()

skipped_bytes = skipped_weights * 2
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

print(f"\nM59 v3 complete. Analysis time: {time.time()-t1:.1f}s")
