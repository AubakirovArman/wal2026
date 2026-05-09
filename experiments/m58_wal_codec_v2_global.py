"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M58: WAL-0 Codec v2 — Global atoms + Global codebook + Bit-packed IDs.

Two-pass architecture:
  Pass 1: Collect samples from all layers, build global atom table.
  Pass 2: Encode all layers with global atoms, build global codebook,
          apply precomputed recon, measure PPL and compression.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 16

K_ATOMS = 128
L_MAX = 2
KMEANS_ITERS = 5
SAMPLE_SIZE_PER_LAYER = 10_000  # Smaller for faster k-means
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ========================================================================
# PASS 1: Build global atom table
# ========================================================================
print("\n=== PASS 1: Building global atom table ===")
t0 = time.time()

global_samples = []
layer_info = []  # (name, shape, device, is_spiky)

for idx, (name, param) in enumerate(list(model.named_parameters())):
    if len(param.shape) != 2:
        layer_info.append((name, param.shape, param.device, True, None))
        continue
    if "embed_tokens" in name or "lm_head" in name:
        layer_info.append((name, param.shape, param.device, True, None))
        continue
    
    w = param.data.float()
    std = (w / w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)).std().item()
    is_spiky = std < SPIKY_THRESHOLD
    
    if not is_spiky:
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        flat_cpu = flat.cpu()
        if flat_cpu.numel() > SAMPLE_SIZE_PER_LAYER:
            idx_samp = torch.randperm(flat_cpu.numel())[:SAMPLE_SIZE_PER_LAYER]
            samp = flat_cpu[idx_samp]
        else:
            samp = flat_cpu
        
        global_samples.append(samp)
        del flat_cpu
        layer_info.append((name, param.shape, param.device, False, std))
    else:
        layer_info.append((name, param.shape, param.device, True, std))

# Build global atoms on GPU
global_samples_cat = torch.cat(global_samples).cuda(2)
print(f"Global samples: {global_samples_cat.numel():,} values")

global_atoms = build_atoms_kmeans(global_samples_cat, K_ATOMS, KMEANS_ITERS, device="cuda:2")
global_atoms = global_atoms.cuda(2)
print(f"Global atoms built in {time.time()-t0:.1f}s")
print(f"Atoms range: [{global_atoms.min():.4f}, {global_atoms.max():.4f}]")

del global_samples_cat, global_samples
torch.cuda.empty_cache()

# ========================================================================
# PASS 2: Encode with global atoms, build global codebook, apply recon
# ========================================================================
print("\n=== PASS 2: Encoding with global atoms + building global codebook ===")

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

all_programs_cpu = []
stats = {"encoded": 0, "skipped": 0, "total_weights": 0}
t0_pass2 = time.time()

for idx, (name, shape, device, is_spiky, std) in enumerate(layer_info):
    if is_spiky:
        stats["skipped"] += 1
        continue
    
    param = dict(model.named_parameters())[name]
    w = param.data.float()
    param_device = param.device
    
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    flat = w_norm.reshape(-1)
    
    # Move atoms to param device if different
    atoms_dev = global_atoms.to(param_device)
    
    # Encode
    indices, signs, recon = wal_encode_scalar_gpu(flat.to(param_device), atoms_dev, L_MAX)
    packed = pack_programs(indices, signs)
    
    all_programs_cpu.append(packed.cpu())
    stats["total_weights"] += packed.numel()
    
    # Apply recon directly (no codebook yet — we'll rebuild with global codebook after)
    w_hat = recon.reshape(w.shape) * row_scale.to(param_device)
    param.data.copy_(w_hat.to(param.dtype))
    
    # Cleanup
    del indices, signs, recon, packed, atoms_dev, flat, w_norm, row_scale, w
    if idx % 20 == 0:
        torch.cuda.empty_cache()
    
    stats["encoded"] += 1
    if stats["encoded"] % 50 == 0:
        print(f"  Encoded {stats['encoded']} params... elapsed={time.time()-t0_pass2:.0f}s")

print(f"\nPass 2 encode done in {time.time()-t0_pass2:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Total weights: {stats['total_weights']:,}")

# ========================================================================
# Build global codebook from all programs
# ========================================================================
print("\n=== Building global codebook ===")
t0 = time.time()

all_packed = torch.cat(all_programs_cpu)
global_unique, global_inverse, global_counts = torch.unique(all_packed, return_inverse=True, return_counts=True)
global_num_unique = global_unique.numel()

print(f"Global unique programs: {global_num_unique:,}")
print(f"Global vs total weights: {global_num_unique / stats['total_weights'] * 100:.4f}%")

# Build precomputed recon table
global_codebook_recon = torch.zeros(global_num_unique, dtype=torch.float32, device="cpu")
atoms_cpu = global_atoms.cpu()
for i in range(global_num_unique):
    a0, s0, a1, s1 = unpack_program(int(global_unique[i].item()))
    global_codebook_recon[i] = atoms_cpu[a0].item() * s0 + atoms_cpu[a1].item() * s1

print(f"Codebook built in {time.time()-t0:.1f}s")

# ========================================================================
# Compression analysis
# ========================================================================
print("\n=== Compression Analysis ===")

# Bits needed for global codebook ID
bits_per_id = torch.log2(torch.tensor(float(global_num_unique))).item()
print(f"Bits per program ID (fixed-width): {bits_per_id:.2f}")

# Huffman entropy
probs = global_counts.float() / global_counts.sum()
entropy = -(probs * torch.log2(probs.clamp_min(1e-10))).sum().item()
print(f"Bits per program ID (Huffman): {entropy:.2f}")

# Total compressed size
# Program IDs + codebook table + atom table
total_weights = stats["total_weights"]
program_bits = entropy * total_weights
codebook_table_bytes = global_num_unique * L_MAX * 2  # atom_id + sign per step
atom_table_bytes = K_ATOMS * 4  # float32 atoms

# Skipped weights (stored as bfloat16 = 2 bytes)
skipped_weights = sum(p.numel() for n, p in model.named_parameters() if any(x[0] == n and x[3] for x in layer_info))
skipped_bytes = skipped_weights * 2

# Overhead: per-layer row scales
encoded_layers = stats["encoded"]
row_scale_overhead = encoded_layers * 8192 * 4  # approx, assume 8192 rows per layer

print(f"\nSize breakdown:")
print(f"  Program IDs (Huffman): {program_bits / 8 / 1e9:.2f} GB")
print(f"  Program IDs (fixed {bits_per_id:.0f}-bit): {total_weights * bits_per_id / 8 / 1e9:.2f} GB")
print(f"  Codebook table: {codebook_table_bytes / 1024:.1f} KB")
print(f"  Atom table: {atom_table_bytes / 1024:.1f} KB")
print(f"  Skipped weights (bf16): {skipped_bytes / 1e9:.2f} GB")
print(f"  Row scales (approx): {row_scale_overhead / 1e9:.2f} GB")

original_size_gb = sum(p.numel() * 2 for p in model.parameters()) / 1e9  # bfloat16 = 2 bytes
compressed_gb = total_weights * bits_per_id / 8 / 1e9 + skipped_bytes / 1e9 + row_scale_overhead / 1e9
print(f"\nOriginal size: {original_size_gb:.2f} GB")
print(f"Compressed size (fixed-width IDs): {compressed_gb:.2f} GB")
print(f"Compression ratio: {original_size_gb / compressed_gb:.2f}x")

# ========================================================================
# PPL Validation
# ========================================================================
print("\n=== PPL Validation ===")
print("Loading WikiText-2...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
num_steps = 0
for begin_loc in range(0, seq_len, stride):
    if num_steps >= max_samples:
        break
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1
    if num_steps % 4 == 0:
        print(f"  Step {num_steps}/{max_samples} loss={outputs.loss.item():.4f}")
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"\nWAL-0 v2 (Global atoms + Global codebook) PPL ({num_steps} steps, {end_loc} tokens): {ppl.item():.4f}")
print(f"Baseline: 2.7805")
print(f"M46: 2.7821")
print(f"M57: 2.7828")

print("\nM58 complete.")
