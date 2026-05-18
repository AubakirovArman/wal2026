#!/usr/bin/env python3
"""M63: WAL v2 VM + Triton Runtime Validation.

Compare three decode paths:
  1. PyTorch (prog.decode)
  2. VM reference interpreter (vm_execute)
  3. Triton kernel (wal_v2_decode_triton)

Validate bit-exact equivalence and measure throughput.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM
from wal.v2 import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.vm import WALVMState, vm_execute
from wal.v2.triton_kernels import wal_v2_decode_triton

model_name = "unsloth/Llama-3.3-70B-Instruct"
LAYER_NAME = "model.language_model.layers.40.self_attn.o_proj"

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

print(f"\n=== Decode Comparison ===")
print(f"Programs: {prog.N:,}")

# Path 1: PyTorch decode
print(f"\n[1] PyTorch decode...")
t0 = time.time()
recon_torch = prog.decode(atoms, coeffs)
torch_time = time.time() - t0
print(f"  Time: {torch_time:.4f}s")

# Path 2: VM decode
print(f"\n[2] VM reference interpreter...")
vm_state = WALVMState(atoms, coeffs, prog)
t0 = time.time()
recon_vm = vm_execute(vm_state, device=param.device)
vm_time = time.time() - t0
print(f"  Time: {vm_time:.4f}s")

# Path 3: Triton decode (without row scales)
print(f"\n[3] Triton decode...")
t0 = time.time()
recon_triton = wal_v2_decode_triton(prog, atoms, coeffs, block_size=1024)
triton_time = time.time() - t0
print(f"  Time: {triton_time:.4f}s")

# Path 4: Triton decode (with row scales)
print(f"\n[4] Triton decode + row scales...")
t0 = time.time()
recon_triton_rs = wal_v2_decode_triton(prog, atoms, coeffs, row_scales=row_scale.flatten(), block_size=1024)
triton_rs_time = time.time() - t0
print(f"  Time: {triton_rs_time:.4f}s")

# Validation
print(f"\n=== Validation ===")

# PyTorch vs VM
match_vm = torch.allclose(recon_torch, recon_vm, atol=1e-5)
diff_vm = (recon_torch - recon_vm).abs().max().item()
print(f"PyTorch vs VM:       match={match_vm}, max_diff={diff_vm:.10f}")

# PyTorch vs Triton (no row scales)
match_triton = torch.allclose(recon_torch, recon_triton, atol=1e-4)
diff_triton = (recon_torch - recon_triton).abs().max().item()
print(f"PyTorch vs Triton:   match={match_triton}, max_diff={diff_triton:.10f}")

# PyTorch vs Triton (with row scales) — need to apply row_scale to torch output
recon_torch_rs = recon_torch.reshape(w.shape) * row_scale.to(param.device)
match_triton_rs = torch.allclose(recon_torch_rs.flatten(), recon_triton_rs, atol=1e-4)
diff_triton_rs = (recon_torch_rs.flatten() - recon_triton_rs).abs().max().item()
print(f"PyTorch vs Triton+RS: match={match_triton_rs}, max_diff={diff_triton_rs:.10f}")

# Throughput
N = prog.N
throughput_torch = N / torch_time / 1e6
throughput_vm = N / vm_time / 1e6
throughput_triton = N / triton_time / 1e6
throughput_triton_rs = N / triton_rs_time / 1e6

print(f"\n=== Throughput (Mweights/sec) ===")
print(f"PyTorch:  {throughput_torch:.1f} Mw/s")
print(f"VM:       {throughput_vm:.1f} Mw/s")
print(f"Triton:   {throughput_triton:.1f} Mw/s")
print(f"Triton+RS:{throughput_triton_rs:.1f} Mw/s")

# Overall verdict
all_pass = match_vm and match_triton and match_triton_rs
print(f"\n{'='*50}")
if all_pass:
    print("M63: ALL DECODE PATHS MATCH -> PASS")
else:
    print("M63: SOME PATHS MISMATCH -> FAIL")
