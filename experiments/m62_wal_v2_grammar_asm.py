#!/usr/bin/env python3
"""M62: WAL v2 Grammar & Assembler round-trip validation.

1. Encode layer 40 o_proj with WAL v2
2. Disassemble to text (full + unique formats)
3. Round-trip assemble back (sample of 10K programs)
4. Verify reconstruction is identical
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.v2 import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.asm import assemble, disassemble

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.layers.40.self_attn.o_proj"

K_ATOMS = 256
C_COEFFS = 16
KMEANS_ITERS = 5
LLOYD_MAX_ITERS = 5
SAMPLE_SIZE = 1_000_000

print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

param = None
for name, p in model.named_parameters():
    if LAYER_NAME in name and len(p.shape) == 2:
        param = p
        break

assert param is not None
print(f"\nTarget: {name}, shape={param.shape}, device={param.device}")

w = param.data.float()
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)

if flat.numel() > SAMPLE_SIZE:
    idx_samp = torch.randperm(flat.numel(), device=param.device)[:SAMPLE_SIZE]
    samples = flat[idx_samp]
else:
    samples = flat

atoms_data = build_atoms_kmeans_v2(samples, K_ATOMS, KMEANS_ITERS, device=param.device)
atoms = AtomTable(atoms_data.to(param.device))

coeffs_data = build_coeff_table(flat, atoms_data, C_COEFFS, LLOYD_MAX_ITERS, device=param.device)
coeffs = CoeffTable(coeffs_data.to(param.device))

prog, recon = wal_encode_v2(flat, atoms, coeffs, residual_threshold=0.0, batch=1_048_576, shape=w.shape)

print(f"\n=== Program Buffer ===")
print(f"Programs: {prog.N:,}")
print(f"Shape: {prog.shape}")
print(f"Unique atom_ids: {prog.atom_ids.unique().numel()}")
print(f"Unique coeff_ids: {prog.coeff_ids.unique().numel()}")

# Test 1: Disassemble first 20 programs
print(f"\n=== Test 1: Text Format (first 20 programs) ===")
text_preview = disassemble(prog, atoms, coeffs, max_programs=20, format="full")
print(text_preview)

# Test 2: Unique program summary
print(f"\n=== Test 2: Unique Program Summary ===")
t0 = time.time()
text_unique = disassemble(prog, atoms, coeffs, format="unique")
print(f"Disassemble time: {time.time()-t0:.1f}s")
lines = text_unique.split('\n')
for line in lines[:25]:
    print(line)
if len(lines) > 25:
    print(f"; ... {len(lines) - 25} more lines ...")

# Test 3: Round-trip on sample of 10K programs
print(f"\n=== Test 3: Round-trip (10K program sample) ===")
sample_n = 10000
t0 = time.time()
text_sample = disassemble(prog, atoms, coeffs, max_programs=sample_n, format="full")
disasm_time = time.time() - t0
print(f"Disassemble {sample_n} programs: {disasm_time:.3f}s")

t0 = time.time()
prog_sample = assemble(text_sample, atoms, coeffs)
asm_time = time.time() - t0
print(f"Assemble {sample_n} programs: {asm_time:.3f}s")

# Verify round-trip
atom_match = (prog.atom_ids[:sample_n] == prog_sample.atom_ids).all().item()
coeff_match = (prog.coeff_ids[:sample_n] == prog_sample.coeff_ids).all().item()
resid_match = (prog.residuals[:sample_n] == prog_sample.residuals).all().item()
hasr_match = (prog.has_residual[:sample_n] == prog_sample.has_residual).all().item()

print(f"\nRound-trip verification:")
print(f"  atom_ids match:     {atom_match}")
print(f"  coeff_ids match:    {coeff_match}")
print(f"  residuals match:    {resid_match}")
print(f"  has_residual match: {hasr_match}")

if atom_match and coeff_match and resid_match and hasr_match:
    print("  -> ROUND-TRIP PASS")
else:
    print("  -> ROUND-TRIP FAIL")
    # Debug: find mismatches
    mismatches = (prog.atom_ids[:sample_n] != prog_sample.atom_ids) | (prog.coeff_ids[:sample_n] != prog_sample.coeff_ids)
    if mismatches.any():
        first_bad = mismatches.nonzero()[0].item()
        print(f"  First mismatch at index {first_bad}")
        print(f"    Original:  atom={prog.atom_ids[first_bad].item()}, coeff={prog.coeff_ids[first_bad].item()}")
        print(f"    Assembled: atom={prog_sample.atom_ids[first_bad].item()}, coeff={prog_sample.coeff_ids[first_bad].item()}")

# Test 4: Reconstruction from assembled sample
print(f"\n=== Test 4: Reconstruction Match ===")
recon_sample = prog_sample.decode(atoms, coeffs)
recon_match = torch.allclose(recon[:sample_n], recon_sample, atol=1e-5)
recon_diff = (recon[:sample_n] - recon_sample).abs().max().item()
print(f"Reconstruction match: {recon_match}")
print(f"Max diff: {recon_diff:.10f}")

print(f"\nM62 complete.")
