"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M54b: WAL-0 codebook packing + direct decode from codebook.

Two decode strategies:
1. Atom-lookup decode: program_id → (atom_ids, signs) → gather atoms → sum
2. Precomputed-recon decode: program_id → recon_value (precomputed float32)

Both GPU-native. Compare speed vs raw WAL-0 decode.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu
from wal.triton_encode import wal_decode_scalar_kernel
import triton

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
print(f"Device: {w.device}")

# Row normalization
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)

# Atoms
samples = flat[torch.randperm(flat.numel(), device=flat.device)[:SAMPLE_SIZE]] if flat.numel() > SAMPLE_SIZE else flat
atoms = build_atoms_kmeans(samples, K_ATOMS, KMEANS_ITERS, device=flat.device).to(flat.device)

# Encode GPU-native
t0 = time.time()
indices, signs, recon = wal_encode_scalar_gpu(flat, atoms, L_MAX)
torch.cuda.synchronize()
print(f"Encode: {time.time()-t0:.3f}s")

# === CODEBOOK BUILDING ===
def pack_programs(indices, signs):
    sign_map = signs.clone()
    sign_map[signs == -1] = 0
    sign_map[signs == 0] = 1
    sign_map[signs == 1] = 2
    packed = (sign_map[:, 0].long() << 16) | (indices[:, 0].long() << 9) | \
             (sign_map[:, 1].long() << 7) | (indices[:, 1].long())
    return packed

def unpack_program(packed_val):
    v = int(packed_val)
    atom_1 = v & 0x7F
    sign_1 = (v >> 7) & 0x3
    atom_0 = (v >> 9) & 0x7F
    sign_0 = (v >> 16) & 0x3
    sign_map_rev = {0: -1, 1: 0, 2: 1}
    return (atom_0, sign_map_rev[sign_0], atom_1, sign_map_rev[sign_1])

packed = pack_programs(indices, signs)
unique_prog, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
num_unique = unique_prog.numel()
print(f"\nCodebook: {num_unique:,} unique programs")

# Sort by frequency for better Huffman coding later
sorted_counts, sort_idx = torch.sort(counts, descending=True)
sorted_progs = unique_prog[sort_idx]
# Create ID mapping: old packed value → new compact ID
program_ids = inverse  # [N] compact IDs 0..num_unique-1, matches unique_prog order

# === DECODE STRATEGY 1: Atom-lookup from codebook ===
# Build codebook tables on GPU
codebook_indices = torch.zeros(num_unique, L_MAX, dtype=torch.uint8, device=flat.device)
codebook_signs = torch.zeros(num_unique, L_MAX, dtype=torch.int8, device=flat.device)
for i, prog_packed in enumerate(sorted_progs):
    a0, s0, a1, s1 = unpack_program(int(prog_packed.item()))
    codebook_indices[i, 0] = a0
    codebook_indices[i, 1] = a1
    codebook_signs[i, 0] = s0
    codebook_signs[i, 1] = s1

# Decode: gather atoms via codebook
t0 = time.time()
gathered = atoms[codebook_indices[program_ids].long()] * codebook_signs[program_ids].float()
recon_v1 = gathered.sum(dim=1)
torch.cuda.synchronize()
t_v1 = time.time() - t0
print(f"Decode v1 (atom-lookup): {t_v1*1000:.2f} ms")

# === DECODE STRATEGY 2: Precomputed recon values ===
codebook_recon = torch.zeros(num_unique, dtype=torch.float32, device=flat.device)
for i in range(num_unique):
    a0, s0, a1, s1 = unpack_program(int(sorted_progs[i].item()))
    codebook_recon[i] = atoms[a0].item() * s0 + atoms[a1].item() * s1

# Decode: simple lookup
t0 = time.time()
recon_v2 = codebook_recon[program_ids]
torch.cuda.synchronize()
t_v2 = time.time() - t0
print(f"Decode v2 (precomputed recon): {t_v2*1000:.2f} ms")

# === DECODE STRATEGY 3: Triton kernel (raw programs) ===
N = flat.numel()
block_size = 1024
grid = (triton.cdiv(N, block_size),)
recon_v3 = torch.empty(N, dtype=torch.float32, device=flat.device)

t0 = time.time()
with torch.cuda.device(flat.device):
    wal_decode_scalar_kernel[grid](
        indices, signs, atoms, recon_v3,
        N, K_ATOMS, L_MAX,
        BLOCK_SIZE=block_size,
    )
torch.cuda.synchronize()
t_v3 = time.time() - t0
print(f"Decode v3 (Triton raw): {t_v3*1000:.2f} ms")

# === VERIFICATION ===
print("\n=== Verification ===")
rel_v1 = ((recon_v1 - recon).abs() / recon.abs().clamp_min(1e-8)).max().item()
rel_v2 = ((recon_v2 - recon).abs() / recon.abs().clamp_min(1e-8)).max().item()
rel_v3 = ((recon_v3 - recon).abs() / recon.abs().clamp_min(1e-8)).max().item()
print(f"Max rel error v1 (atom-lookup):  {rel_v1:.2e}")
print(f"Max rel error v2 (precomputed):  {rel_v2:.2e}")
print(f"Max rel error v3 (Triton raw):   {rel_v3:.2e}")

# Throughput comparison
mw = flat.numel() / 1e6
print(f"\nThroughput ({mw:.1f}M weights):")
print(f"  v1 atom-lookup:     {mw/t_v1:.1f} Mw/s")
print(f"  v2 precomputed:     {mw/t_v2:.1f} Mw/s")
print(f"  v3 Triton raw:      {mw/t_v3:.1f} Mw/s")

# Compression analysis
print(f"\n=== Compression ===")
print(f"Raw (indices+signs):     {2 * 8:.0f} bits/weight")
print(f"Codebook ID (uint16):    {16:.0f} bits/weight")
print(f"Codebook ID (uint10):    {10:.0f} bits/weight (fits in 1079 unique)")
print(f"Codebook table:          {num_unique * L_MAX * 2} bytes")
print(f"Precomputed recon table: {num_unique * 4} bytes")

print("\nM54b complete.")
