"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M87: KV-cache WAL Decode Speed Benchmark.

Measure decode throughput for WAL-encoded KV-cache.
Compare: CPU decode, GPU decode, and precomputed lookup.
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import time

print("=" * 60)
print("M87: KV-cache Decode Speed Benchmark")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.decoder import wal_decode_v1
from wal.v1.isa import AtomTableV1, AtomDef

# Simulate KV-cache for one layer
# Shape: [batch=1, num_kv_heads=8, seq_len=8192, head_dim=128]
seq_len = 8192
shape = (1, 8, seq_len, 128)
N = 1 * 8 * seq_len * 128

print(f"\nSimulated KV-cache: {shape}")
print(f"Total elements: {N:,}")

# Generate synthetic KV-cache (smoother than random, to match real KV distribution)
k_raw = torch.randn(shape) * 1.8  # Match K std from M84
v_raw = torch.randn(shape) * 0.2  # Match V std from M84

# ---- K-cache decode benchmark ----
print("\n" + "=" * 60)
print("K-cache Decode (K=256, C=16)")
print("=" * 60)

k_flat = k_raw.reshape(-1)
k_atoms = build_l0_atoms(k_flat, K=256, iters=3)
k_coeffs = build_coeff_table(k_flat, k_atoms, C=16, iters=3)
k_prog, _ = wal_encode_v1(k_flat, k_atoms, k_coeffs)

k_defs = [AtomDef(level=0, op="CONST") for _ in range(256)]
k_table = AtomTableV1(k_atoms, k_defs)

# CPU benchmark
if device.type == "cuda":
    # Warmup
    for _ in range(3):
        _ = wal_decode_v1(k_prog, k_table, k_coeffs)
    
    cpu_start = time.time()
    iterations = 10
    for _ in range(iterations):
        k_dec = wal_decode_v1(k_prog, k_table, k_coeffs)
    cpu_elapsed = time.time() - cpu_start
    cpu_total = N * iterations
    cpu_speed = cpu_total / cpu_elapsed / 1e6
    print(f"  CPU: {cpu_total:,} elements in {cpu_elapsed:.3f}s = {cpu_speed:.1f} Mw/s")
    
    # GPU benchmark
    k_prog_gpu = k_prog
    k_prog_gpu.atom_ids = k_prog.atom_ids.to(device)
    k_prog_gpu.coeff_ids = k_prog.coeff_ids.to(device)
    k_table_gpu = AtomTableV1(k_atoms.to(device), k_defs)
    k_coeffs_gpu = k_coeffs.to(device)
    
    torch.cuda.synchronize()
    gpu_start = time.time()
    for _ in range(iterations):
        k_dec = wal_decode_v1(k_prog_gpu, k_table_gpu, k_coeffs_gpu)
    torch.cuda.synchronize()
    gpu_elapsed = time.time() - gpu_start
    gpu_total = N * iterations
    gpu_speed = gpu_total / gpu_elapsed / 1e6
    print(f"  GPU: {gpu_total:,} elements in {gpu_elapsed:.3f}s = {gpu_speed:.1f} Mw/s")
else:
    # CPU only
    cpu_start = time.time()
    iterations = 10
    for _ in range(iterations):
        k_dec = wal_decode_v1(k_prog, k_table, k_coeffs)
    cpu_elapsed = time.time() - cpu_start
    cpu_total = N * iterations
    cpu_speed = cpu_total / cpu_elapsed / 1e6
    print(f"  CPU: {cpu_total:,} elements in {cpu_elapsed:.3f}s = {cpu_speed:.1f} Mw/s")

# ---- V-cache decode benchmark ----
print("\n" + "=" * 60)
print("V-cache Decode (K=64, C=8)")
print("=" * 60)

v_flat = v_raw.reshape(-1)
v_atoms = build_l0_atoms(v_flat, K=64, iters=3)
v_coeffs = build_coeff_table(v_flat, v_atoms, C=8, iters=3)
v_prog, _ = wal_encode_v1(v_flat, v_atoms, v_coeffs)

v_defs = [AtomDef(level=0, op="CONST") for _ in range(64)]
v_table = AtomTableV1(v_atoms, v_defs)

if device.type == "cuda":
    # CPU
    cpu_start = time.time()
    for _ in range(iterations):
        v_dec = wal_decode_v1(v_prog, v_table, v_coeffs)
    cpu_elapsed = time.time() - cpu_start
    cpu_speed = cpu_total / cpu_elapsed / 1e6
    print(f"  CPU: {cpu_total:,} elements in {cpu_elapsed:.3f}s = {cpu_speed:.1f} Mw/s")
    
    # GPU
    v_prog_gpu = v_prog
    v_prog_gpu.atom_ids = v_prog.atom_ids.to(device)
    v_prog_gpu.coeff_ids = v_prog.coeff_ids.to(device)
    v_table_gpu = AtomTableV1(v_atoms.to(device), v_defs)
    v_coeffs_gpu = v_coeffs.to(device)
    
    torch.cuda.synchronize()
    gpu_start = time.time()
    for _ in range(iterations):
        v_dec = wal_decode_v1(v_prog_gpu, v_table_gpu, v_coeffs_gpu)
    torch.cuda.synchronize()
    gpu_elapsed = time.time() - gpu_start
    gpu_total = N * iterations
    gpu_speed = gpu_total / gpu_elapsed / 1e6
    print(f"  GPU: {gpu_total:,} elements in {gpu_elapsed:.3f}s = {gpu_speed:.1f} Mw/s")
else:
    cpu_start = time.time()
    for _ in range(iterations):
        v_dec = wal_decode_v1(v_prog, v_table, v_coeffs)
    cpu_elapsed = time.time() - cpu_start
    cpu_speed = cpu_total / cpu_elapsed / 1e6
    print(f"  CPU: {cpu_total:,} elements in {cpu_elapsed:.3f}s = {cpu_speed:.1f} Mw/s")

# ---- Precomputed lookup benchmark ----
print("\n" + "=" * 60)
print("Precomputed Lookup (fast path)")
print("=" * 60)

# For precomputed lookup, decode = single index_select
if device.type == "cuda":
    flat_atoms = torch.tensor([k_table.resolve(i) for i in range(k_table.K_total)], dtype=torch.float32, device=device)
    
    torch.cuda.synchronize()
    lookup_start = time.time()
    for _ in range(iterations):
        out = flat_atoms[k_prog_gpu.atom_ids.long()] * k_coeffs_gpu[k_prog_gpu.coeff_ids.long()]
    torch.cuda.synchronize()
    lookup_elapsed = time.time() - lookup_start
    lookup_speed = gpu_total / lookup_elapsed / 1e6
    print(f"  GPU lookup: {gpu_total:,} elements in {lookup_elapsed:.3f}s = {lookup_speed:.1f} Mw/s")

# ---- Real-world scenario: full model KV-cache ----
print("\n" + "=" * 60)
print("Full Model KV-cache (80 layers, 8K context)")
print("=" * 60)

layers = 80
seq_len = 8192
elements_per_layer = 8 * seq_len * 128 * 2  # K + V
total_elements = layers * elements_per_layer
print(f"Total KV elements: {total_elements:,}")
print(f"At bf16: {total_elements * 2 / 1024**3:.2f} GB")
print(f"At 12-bit WAL: {total_elements * 12 / 8 / 1024**3:.2f} GB")
print(f"At 10-bit WAL: {total_elements * 10 / 8 / 1024**3:.2f} GB")
print(f"At 8-bit WAL: {total_elements * 8 / 8 / 1024**3:.2f} GB")

if device.type == "cuda":
    # Estimate full decode time
    decode_time_per_layer = gpu_elapsed / iterations
    total_decode_time = layers * decode_time_per_layer
    print(f"\nEstimated full decode time: {total_decode_time:.3f}s")
    print(f"Decode overhead per token: {total_decode_time * 1000:.1f} ms")

print("\n" + "=" * 60)
print("M87: COMPLETE")
print("=" * 60)
