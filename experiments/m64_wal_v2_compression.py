"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M64: WAL v2 Binary Format Round-Trip Validation.

1. Encode layer 40 o_proj with WAL v2
2. Serialize to binary format v0.1
3. Deserialize back
4. Verify reconstruction is identical
5. Measure compression ratio
"""
import torch
import time
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.v2 import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.format import serialize_wal_v2, deserialize_wal_v2

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

print(f"\n=== Original ===")
print(f"Programs: {prog.N:,}")
print(f"Shape: {prog.shape}")
print(f"Has residuals: {prog.has_residual.sum().item()}")

# Serialize
print(f"\n=== Serialize ===")
t0 = time.time()
binary = serialize_wal_v2(
    prog, atoms, coeffs,
    row_scales=row_scale.flatten(),
    metadata={'parameter': name, 'dtype': 'bfloat16'},
)
serialize_time = time.time() - t0
print(f"Serialize time: {serialize_time:.3f}s")
print(f"Binary size: {len(binary):,} bytes ({len(binary)/1e6:.2f} MB)")

# Deserialize
print(f"\n=== Deserialize ===")
t0 = time.time()
prog2, atoms2, coeffs2, row_scales2, meta = deserialize_wal_v2(binary)
deserialize_time = time.time() - t0
print(f"Deserialize time: {deserialize_time:.3f}s")
print(f"Metadata: {meta}")

# Verify round-trip
print(f"\n=== Round-Trip Validation ===")
recon2 = prog2.decode(atoms2, coeffs2)
recon2_scaled = recon2.to(param.device).reshape(w.shape) * row_scales2.to(param.device)

# Compare to original
match_atom = (prog.atom_ids == prog2.atom_ids).all().item()
match_coeff = (prog.coeff_ids == prog2.coeff_ids).all().item()
match_resid = (prog.residuals == prog2.residuals).all().item()
match_hasr = (prog.has_residual == prog2.has_residual).all().item()
match_atoms = torch.allclose(atoms.values.cpu(), atoms2.values)
match_coeffs = torch.allclose(coeffs.values.cpu(), coeffs2.values)
match_row = torch.allclose(row_scale.flatten().cpu(), row_scales2)
match_recon = torch.allclose(recon, recon2.to(param.device), atol=1e-5)

print(f"atom_ids match:    {match_atom}")
print(f"coeff_ids match:   {match_coeff}")
print(f"residuals match:   {match_resid}")
print(f"has_residual match:{match_hasr}")
print(f"atoms match:       {match_atoms}")
print(f"coeffs match:      {match_coeffs}")
print(f"row_scales match:  {match_row}")
print(f"reconstruction match: {match_recon}")

all_pass = (match_atom and match_coeff and match_resid and match_hasr and
            match_atoms and match_coeffs and match_row and match_recon)

# Compression analysis
N = prog.N
original_bytes = N * 2  # bf16
prog_bytes = N * 1.5  # 12 bits/weight packed
atom_bytes = K_ATOMS * 4
coeff_bytes = C_COEFFS * 4
row_scale_bytes = param.shape[0] * 4
meta_bytes = len(json.dumps(meta).encode())

theoretical_size = prog_bytes + atom_bytes + coeff_bytes + row_scale_bytes + meta_bytes

print(f"\n=== Compression Analysis ===")
print(f"Original (bf16):     {original_bytes/1e6:.2f} MB")
print(f"WAL v2 binary:       {len(binary)/1e6:.2f} MB")
print(f"Theoretical minimum: {theoretical_size/1e6:.2f} MB")
print(f"Binary overhead:     {(len(binary) - theoretical_size)/1e3:.1f} KB")
print(f"Compression ratio:   {original_bytes / len(binary):.2f}x")
print(f"Bits/weight:         {len(binary) * 8 / N:.1f}")

print(f"\n{'='*50}")
if all_pass:
    print("M64: BINARY ROUND-TRIP PASS")
else:
    print("M64: BINARY ROUND-TRIP FAIL")
