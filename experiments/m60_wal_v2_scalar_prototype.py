#!/usr/bin/env python3
"""M60: WAL v2 scalar prototype — single layer validation.

Compare WAL v2 (single-call + continuous coefficients) vs WAL-0 baseline
on layer 40 o_proj of Llama 3.3 70B.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.v2 import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj"

# WAL v2 config
K_ATOMS = 256
C_COEFFS = 16
KMEANS_ITERS = 10
LLOYD_MAX_ITERS = 10
SAMPLE_SIZE = 2_000_000
RESIDUAL_THRESHOLD = 0.0  # Start without residuals

print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

# Find target parameter
param = None
param_name = None
for name, p in model.named_parameters():
    if LAYER_NAME in name and len(p.shape) == 2:
        param = p
        param_name = name
        break

assert param is not None, f"Parameter {LAYER_NAME} not found"
print(f"\nTarget: {param_name}, shape={param.shape}, device={param.device}")

# Row normalize
w = param.data.float()
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
flat = w_norm.reshape(-1)
print(f"  Total weights: {flat.numel():,}")

# Sample for atom building
if flat.numel() > SAMPLE_SIZE:
    idx_samp = torch.randperm(flat.numel(), device=param.device)[:SAMPLE_SIZE]
    samples = flat[idx_samp]
else:
    samples = flat

print(f"\nBuilding atom table (K={K_ATOMS}, iters={KMEANS_ITERS})...")
t0 = time.time()
atoms_data = build_atoms_kmeans_v2(samples, K_ATOMS, KMEANS_ITERS, device=param.device)
atoms = AtomTable(atoms_data)
print(f"  Done in {time.time()-t0:.1f}s")

print(f"\nBuilding coeff table (C={C_COEFFS}, iters={LLOYD_MAX_ITERS})...")
t0 = time.time()
coeffs_data = build_coeff_table(flat, atoms_data, C_COEFFS, LLOYD_MAX_ITERS, device=param.device)
coeffs = CoeffTable(coeffs_data)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Coeff values: {coeffs_data[:8].tolist()} ...")

print(f"\nEncoding all weights...")
t0 = time.time()
prog, recon = wal_encode_v2(flat, atoms, coeffs, residual_threshold=RESIDUAL_THRESHOLD, batch=1_048_576)
encode_time = time.time() - t0
print(f"  Done in {encode_time:.1f}s")

# Apply row scale
recon_scaled = recon.to(param.device).reshape(w.shape) * row_scale

# Metrics
relMSE = ((recon_scaled - w) ** 2).sum() / (w ** 2).sum()
relMAE = (recon_scaled - w).abs().mean() / w.abs().mean()
max_err = (recon_scaled - w).abs().max()

print(f"\n=== WAL v2 Quality ===")
print(f"relMSE:  {relMSE.item():.8f}")
print(f"relMAE:  {relMAE.item():.8f}")
print(f"max_err: {max_err.item():.6f}")

# Compare to dense baseline (copy original back)
param.data.copy_(w.to(param.dtype))

# Forward pass through layer to measure output error
model.eval()
with torch.no_grad():
    # Create dummy input
    dummy_input = torch.randn(1, param.shape[1], dtype=torch.bfloat16, device=param.device)
    
    # Dense output
    dense_out = torch.matmul(dummy_input, param.data.T)
    
    # WAL output
    param.data.copy_(recon_scaled.to(param.dtype))
    wal_out = torch.matmul(dummy_input, param.data.T)
    
    # Restore dense
    param.data.copy_(w.to(param.dtype))
    
    output_relMSE = ((wal_out - dense_out) ** 2).sum() / (dense_out ** 2).sum()
    output_corr = torch.corrcoef(torch.stack([wal_out.flatten(), dense_out.flatten()]))[0, 1]

print(f"\n=== Output Quality ===")
print(f"Output relMSE: {output_relMSE.item():.8f}")
print(f"Output corr:   {output_corr.item():.8f}")

# Compression estimate
N = flat.numel()
bits_per_prog = 8 + 4  # atom_id (8b) + coeff_id (4b) = 12 bits
prog_bytes = N * bits_per_prog / 8
atom_bytes = K_ATOMS * 4
coeff_bytes = C_COEFFS * 4
row_scale_bytes = param.shape[0] * 4
original_bytes = N * 2  # bf16

print(f"\n=== Compression ===")
print(f"Original:      {original_bytes / 1e6:.2f} MB")
print(f"Programs:      {prog_bytes / 1e6:.2f} MB ({bits_per_prog:.1f} bits/weight)")
print(f"Atom table:    {atom_bytes / 1e3:.2f} KB")
print(f"Coeff table:   {coeff_bytes / 1e3:.2f} KB")
print(f"Row scales:    {row_scale_bytes / 1e3:.2f} KB")
print(f"Total:         {(prog_bytes + atom_bytes + coeff_bytes + row_scale_bytes) / 1e6:.2f} MB")
print(f"Ratio:         {original_bytes / (prog_bytes + atom_bytes + coeff_bytes + row_scale_bytes):.2f}x")

# Program statistics
unique_programs = len(torch.unique(torch.stack([prog.atom_ids, prog.coeff_ids], dim=1), dim=0))
print(f"\n=== Program Stats ===")
print(f"Unique programs: {unique_programs:,} / {N:,} ({unique_programs/N*100:.4f}%)")
print(f"Residuals used:  {prog.has_residual.sum().item():,}")

print(f"\n=== Comparison to WAL-0 Baseline ===")
print(f"WAL-0 (M48):     relMSE=0.00000454, output_relMSE=0.00001574")
print(f"WAL v2 (M60):    relMSE={relMSE.item():.8f}, output_relMSE={output_relMSE.item():.8f}")
if output_relMSE.item() < 0.0001:
    print("  -> QUALITY OK: output relMSE < 1e-4")
else:
    print("  -> WARNING: output relMSE > 1e-4")
