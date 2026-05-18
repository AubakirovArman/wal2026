#!/usr/bin/env python3
"""Debug: find exact program where codebook recon differs from raw recon."""
import torch
from transformers import AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

K_ATOMS = 128
L_MAX = 2
SAMPLE_SIZE = 1_000_000

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

# Test layer 0 o_proj
name = "model.language_model.layers.0.self_attn.o_proj.weight"
param = dict(model.named_parameters())[name]
w = param.data.float()
param_device = param.device
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)

samples = flat[torch.randperm(flat.numel(), device=param_device)[:SAMPLE_SIZE]] if flat.numel() > SAMPLE_SIZE else flat
atoms = build_atoms_kmeans(samples, K_ATOMS, 5, device=param_device).to(param_device)

indices, signs, recon = wal_encode_scalar_gpu(flat.to(param_device), atoms, L_MAX)

packed = pack_programs(indices, signs)
unique_prog, inverse, counts = torch.unique(packed, return_inverse=True, return_counts=True)
num_unique = unique_prog.numel()

codebook_recon = torch.zeros(num_unique, dtype=torch.float32, device=param_device)
for i in range(num_unique):
    a0, s0, a1, s1 = unpack_program(int(unique_prog[i].item()))
    codebook_recon[i] = atoms[a0].item() * s0 + atoms[a1].item() * s1

old_to_new = torch.zeros(unique_prog.max().item() + 1, dtype=torch.int64, device=param_device)
sort_idx = torch.argsort(counts, descending=True)
old_to_new[unique_prog[sort_idx]] = torch.arange(num_unique, device=param_device)
program_ids = old_to_new[packed]
recon_from_codebook = codebook_recon[program_ids]

# Find max diff
diff = (recon_from_codebook - recon).abs()
max_idx = diff.argmax()
print(f"Max diff index: {max_idx.item()}")
print(f"Max diff value: {diff[max_idx].item():.6f}")
print(f"Raw recon: {recon[max_idx].item():.6f}")
print(f"Codebook recon: {recon_from_codebook[max_idx].item():.6f}")
print(f"Program packed: {packed[max_idx].item()}")
print(f"Program ID: {program_ids[max_idx].item()}")

a0, s0, a1, s1 = unpack_program(int(packed[max_idx].item()))
print(f"Unpacked: ({a0}, {s0}) + ({a1}, {s1})")
print(f"Atom[{a0}] = {atoms[a0].item():.6f}, sign={s0}")
print(f"Atom[{a1}] = {atoms[a1].item():.6f}, sign={s1}")
print(f"Manual recon = {atoms[a0].item() * s0 + atoms[a1].item() * s1:.6f}")

# Check if the packed value is in unique_prog correctly
print(f"\nIs packed in unique_prog? {(packed[max_idx] == unique_prog).any().item()}")
print(f"unique_prog contains {packed[max_idx].item()} at index {(unique_prog == packed[max_idx]).nonzero(as_tuple=True)[0].item() if (unique_prog == packed[max_idx]).any() else 'NOT FOUND'}")

# Check codebook_recon for this program
prog_packed_val = packed[max_idx].item()
prog_id = program_ids[max_idx].item()
print(f"codebook_recon[{prog_id}] = {codebook_recon[prog_id].item():.6f}")

# Verify that unique_prog[inverse] reconstructs packed
reconstructed_packed = unique_prog[inverse[max_idx]]
print(f"unique_prog[inverse[{max_idx}]] = {reconstructed_packed.item()}")
print(f"Match: {reconstructed_packed.item() == prog_packed_val}")

# Verify old_to_new mapping
print(f"old_to_new[{prog_packed_val}] = {old_to_new[prog_packed_val].item()}")
print(f"program_ids[{max_idx}] = {program_ids[max_idx].item()}")
