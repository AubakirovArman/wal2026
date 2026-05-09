"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M56a: WAL-0 grammar analysis — structure in the program stream.

Analyze n-grams, spatial correlation, and predictability of program sequences.
All GPU-native.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

LAYER_NAME = "model.layers.40.self_attn.o_proj.weight"
K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000

print("Loading model...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

w = dict(model.named_parameters())[LAYER_NAME].data.float()
print(f"\nLayer: {LAYER_NAME}")
print(f"Shape: {w.shape}, elements: {w.numel():,}")

row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)

samples = flat[torch.randperm(flat.numel(), device=flat.device)[:SAMPLE_SIZE]] if flat.numel() > SAMPLE_SIZE else flat
atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=flat.device).to(flat.device)

# Encode
indices, signs, recon = wal_encode_scalar_gpu(flat, atoms, L_MAX)

# Pack programs to IDs
def pack_programs(indices, signs):
    sign_map = signs.clone()
    sign_map[signs == -1] = 0
    sign_map[signs == 0] = 1
    sign_map[signs == 1] = 2
    packed = (sign_map[:, 0].long() << 16) | (indices[:, 0].long() << 9) | \
             (sign_map[:, 1].long() << 7) | (indices[:, 1].long())
    return packed

packed = pack_programs(indices, signs)
unique_prog, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
num_unique = unique_prog.numel()
N = packed.numel()

# Compact program IDs (sorted by frequency)
sorted_counts, sort_idx = torch.sort(counts, descending=True)
sorted_progs = unique_prog[sort_idx]
old_to_new = torch.zeros(unique_prog.max().item() + 1, dtype=torch.int64, device=packed.device)
old_to_new[sorted_progs] = torch.arange(num_unique, device=packed.device)
program_ids = old_to_new[packed]  # [N] compact IDs

print(f"\n=== Grammar Analysis ===")
print(f"Total programs: {N:,}")
print(f"Unique programs: {num_unique:,}")

# 1. Unigram entropy (already known)
probs = counts.float() / counts.sum()
H1 = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
print(f"\n1. Unigram entropy: {H1:.3f} bits (max: {torch.log2(torch.tensor(float(num_unique))).item():.3f})")

# 2. Bigram entropy
t0 = time.time()
# Build bigram pairs: (prog[i], prog[i+1])
# For a matrix of shape [H, W], neighbors can be row-wise or column-wise
# Let's do both: row-neighbors and column-neighbors
H, W = w.shape

# Row-wise bigrams (adjacent in a row)
row_ids = program_ids.reshape(H, W)
row_bigrams_a = row_ids[:, :-1].reshape(-1)
row_bigrams_b = row_ids[:, 1:].reshape(-1)

# Column-wise bigrams (adjacent in a column)
col_bigrams_a = row_ids[:-1, :].reshape(-1)
col_bigrams_b = row_ids[1:, :].reshape(-1)

# Combine all spatial neighbors
all_a = torch.cat([row_bigrams_a, col_bigrams_a])
all_b = torch.cat([row_bigrams_b, col_bigrams_b])

# Count unique bigrams using packing: pack two uint16 into uint32
# But program_ids can be up to num_unique-1 (~1000), so 10 bits each. Pack into 20 bits.
bigram_packed = (all_a.long() << 10) | all_b.long()
unique_bigrams, bigram_counts = torch.unique(bigram_packed, return_counts=True)
num_bigrams = unique_bigrams.numel()

bigram_probs = bigram_counts.float() / bigram_counts.sum()
H2_joint = -(bigram_probs * torch.log2(bigram_probs.clamp_min(1e-10))).sum().item()
H2_cond = H2_joint - H1  # H(B|A) = H(A,B) - H(A)
print(f"2. Bigram joint entropy: {H2_joint:.3f} bits")
print(f"   Conditional entropy H(prog[i+1]|prog[i]): {H2_cond:.3f} bits")
print(f"   Predictability gain: {H1 - H2_cond:.3f} bits ({(H1 - H2_cond)/H1 * 100:.1f}%)")
print(f"   Unique bigrams: {num_bigrams:,} / {num_unique**2:,} possible ({num_bigrams/num_unique**2 * 100:.4f}%)")

# 3. Spatial autocorrelation (program IDs)
# Compute correlation between program and its neighbors
prog_float = program_ids.float()
row_ids_f = prog_float.reshape(H, W)

# Row neighbor correlation
row_corr_num = ((row_ids_f[:, :-1] - row_ids_f[:, :-1].mean()) * (row_ids_f[:, 1:] - row_ids_f[:, 1:].mean())).sum()
row_corr_den = (((row_ids_f[:, :-1] - row_ids_f[:, :-1].mean())**2).sum() * ((row_ids_f[:, 1:] - row_ids_f[:, 1:].mean())**2).sum()).sqrt()
row_corr = (row_corr_num / row_corr_den).item() if row_corr_den > 0 else 0.0

# Column neighbor correlation
col_corr_num = ((row_ids_f[:-1, :] - row_ids_f[:-1, :].mean()) * (row_ids_f[1:, :] - row_ids_f[1:, :].mean())).sum()
col_corr_den = (((row_ids_f[:-1, :] - row_ids_f[:-1, :].mean())**2).sum() * ((row_ids_f[1:, :] - row_ids_f[1:, :].mean())**2).sum()).sqrt()
col_corr = (col_corr_num / col_corr_den).item() if col_corr_den > 0 else 0.0

print(f"\n3. Spatial autocorrelation:")
print(f"   Row neighbor correlation: {row_corr:.4f}")
print(f"   Col neighbor correlation: {col_corr:.4f}")

# 4. Repeat rate (same program as previous)
row_same = (row_ids[:, :-1] == row_ids[:, 1:]).float().mean().item()
col_same = (row_ids[:-1, :] == row_ids[1:, :]).float().mean().item()
print(f"\n4. Repeat rate:")
print(f"   Row repeat rate: {row_same:.4f} ({row_same*100:.2f}%)")
print(f"   Col repeat rate: {col_same:.4f} ({col_same*100:.2f}%)")

# 5. Most common bigrams
sorted_bcounts, sorted_bidx = torch.sort(bigram_counts, descending=True)
print(f"\n5. Top 10 bigrams:")
for i in range(min(10, num_bigrams)):
    packed_val = unique_bigrams[sorted_bidx[i]]
    count = sorted_bcounts[i].item()
    a_id = (packed_val >> 10).item()
    b_id = (packed_val & 0x3FF).item()
    print(f"   ({a_id:4d} → {b_id:4d}): count={count:8,} ({count/bigram_counts.sum()*100:.3f}%)")

# 6. Entropy rate estimate
# If stream is i.i.d., entropy rate = H1. If there is structure, entropy rate < H1.
# Using Lempel-Ziv style estimate: compressibility
print(f"\n6. Structure summary:")
print(f"   If programs were i.i.d.: entropy rate ≈ {H1:.3f} bits")
print(f"   Estimated entropy rate (bigram): ≈ {H2_cond:.3f} bits")
print(f"   Structure found: {'YES' if H2_cond < H1 * 0.95 else 'WEAK' if H2_cond < H1 * 0.99 else 'NO'}")
print(f"   Grammar potential: {'HIGH' if H2_cond < H1 * 0.8 else 'MODERATE' if H2_cond < H1 * 0.9 else 'LOW'}")

print(f"\nAnalysis time: {time.time()-t0:.3f}s")
print("\nM56a complete.")
