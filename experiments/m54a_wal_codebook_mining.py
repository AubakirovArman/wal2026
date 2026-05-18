#!/usr/bin/env python3
"""M54a: GPU-native WAL-0 program codebook mining on real 70B weights.

Goal: encode a full layer, find unique programs, measure vocabulary and entropy.
Everything stays on GPU.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

# Config
LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj.weight"
K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000

print("Loading model (single layer access)...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

# Get target weight
w = dict(model.named_parameters())[LAYER_NAME].data.float()
print(f"\nLayer: {LAYER_NAME}")
print(f"Shape: {w.shape}, elements: {w.numel():,}")
print(f"Device: {w.device}")

# Row normalization
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)
print(f"Flat shape: {flat.shape}")

# Sample for k-means
if flat.numel() > SAMPLE_SIZE:
    idx_samp = torch.randperm(flat.numel(), device=flat.device)[:SAMPLE_SIZE]
    samples = flat[idx_samp]
else:
    samples = flat

print(f"Samples for k-means: {samples.numel():,}")

# Build atoms on GPU
t0 = time.time()
atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=flat.device)
atoms = atoms.to(flat.device)
print(f"K-means done in {time.time()-t0:.1f}s")
print(f"Atoms range: [{atoms.min():.4f}, {atoms.max():.4f}]")

# Encode — fully GPU-native
t0 = time.time()
indices, signs, recon = wal_encode_scalar_gpu(flat, atoms, L_MAX)
torch.cuda.synchronize()
encode_time = time.time() - t0
print(f"Encode done in {encode_time:.1f}s")

# Verify reconstruction
recon_2d = recon.reshape(w.shape) * row_scale
rel_err = ((recon_2d - w).abs() / w.abs().clamp_min(1e-8)).mean().item()
print(f"Weight relMAE: {rel_err:.6f}")

# === CODEBOOK MINING — ALL ON GPU ===
print("\n=== Codebook Mining (GPU-native) ===")

# Each program is a tuple: (atom_id_0, sign_0, atom_id_1, sign_1)
# Pack into a single int64 for uniqueness search
# atom_id: 7 bits (0-127), sign: 2 bits (-1,0,1 → map to 0,1,2)
# Total per step: 9 bits. For 2 steps: 18 bits. Fits in int32.

def pack_programs(indices, signs):
    """Pack (uint8 atom_id, int8 sign) pairs into int32 program IDs."""
    # Map signs: -1→0, 0→1, +1→2
    sign_map = signs.clone()
    sign_map[signs == -1] = 0
    sign_map[signs == 0] = 1
    sign_map[signs == 1] = 2
    
    # Pack: [sign_0(2bits), atom_0(7bits), sign_1(2bits), atom_1(7bits)]
    # Layout: bits 0-6 = atom_1, bit 7-8 = sign_1, bits 9-15 = atom_0, bits 16-17 = sign_0
    packed = (sign_map[:, 0].long() << 16) | (indices[:, 0].long() << 9) | \
             (sign_map[:, 1].long() << 7) | (indices[:, 1].long())
    return packed

t0 = time.time()
packed = pack_programs(indices, signs)
torch.cuda.synchronize()

# Unique programs and counts
unique_prog, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
torch.cuda.synchronize()
mining_time = time.time() - t0

num_unique = unique_prog.numel()
total_weights = packed.numel()

print(f"Unique programs: {num_unique:,}")
print(f"Total weights: {total_weights:,}")
print(f"Vocabulary ratio: {num_unique/total_weights:.4%}")
print(f"Mining time: {mining_time:.3f}s")

# Entropy
probs = counts.float() / counts.sum()
entropy = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
max_entropy = torch.log2(torch.tensor(float(num_unique))).item()
print(f"\nEntropy: {entropy:.2f} bits (max: {max_entropy:.2f})")
print(f"Effective bits per program: {entropy:.2f}")

# Top-K coverage
sorted_counts, sorted_idx = torch.sort(counts, descending=True)
cumsum = sorted_counts.cumsum(dim=0)
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    if k <= num_unique:
        cov = cumsum[k-1].item() / total_weights
        print(f"  Top-{k:4d} coverage: {cov:.4%}")

# Decode a few top programs
def unpack_program(packed_val):
    v = int(packed_val.item())
    atom_1 = v & 0x7F
    sign_1 = (v >> 7) & 0x3
    atom_0 = (v >> 9) & 0x7F
    sign_0 = (v >> 16) & 0x3
    # Unmap signs
    sign_map_rev = {0: -1, 1: 0, 2: 1}
    return (atom_0, sign_map_rev[sign_0], atom_1, sign_map_rev[sign_1])

print("\nTop 10 most frequent programs:")
for i in range(min(10, num_unique)):
    prog_id = unique_prog[sorted_idx[i]]
    count = sorted_counts[i].item()
    a0, s0, a1, s1 = unpack_program(prog_id)
    # Compute what this program reconstructs
    recon_val = atoms[a0].item() * s0 + atoms[a1].item() * s1
    print(f"  rank={i+1:2d}: count={count:8,} ({count/total_weights:.4%}) | "
          f"({a0:3d},{s0:+2d}) + ({a1:3d},{s1:+2d}) → recon={recon_val:+.6f}")

# Estimate compression
# Raw: 2 bytes/weight (2 x uint8 for atom_id, 2 x int8 for sign = 4 bytes?)
# Actually current: indices uint8 + signs int8 = 2 bytes/weight for lmax=2
# With codebook: store program_id + codebook
bits_per_id = entropy  # ~ if Huffman-coded
bits_program_id = torch.log2(torch.tensor(float(num_unique))).item()
print(f"\nCompression estimate:")
print(f"  Raw (indices+signs): {2 * 8:.1f} bits/weight")
print(f"  Codebook ID (ideal Huffman): {bits_per_id:.2f} bits/weight")
print(f"  Codebook ID (fixed-width): {bits_program_id:.2f} bits/weight")
print(f"  Codebook table size: {num_unique} programs × {L_MAX} steps × (1 byte atom + 1 byte sign) = {num_unique * L_MAX * 2:,} bytes")

print("\nM54a complete.")
