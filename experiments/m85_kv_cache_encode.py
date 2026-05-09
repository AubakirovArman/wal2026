#!/usr/bin/env python3
"""M85: KV-cache WAL Encoding Prototype.

Encode KV-cache with WAL and measure quality.
Test hypotheses from M84:
1. V-cache compresses better than K-cache (lower entropy)
2. Delta-encoding exploits temporal correlation
3. Per-head atoms beat global atoms
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import numpy as np
import time

print("=" * 60)
print("M85: KV-cache WAL Encoding Prototype")
print("=" * 60)

# ---- Setup ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (visible GPUs: {torch.cuda.device_count()})")

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.decoder import wal_decode_v1
from wal.v1.isa import AtomTableV1, AtomDef

print("\nLoading Llama 3.3 70B...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    dtype=torch.bfloat16,
    device_map={"": device},
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct")

# Generate longer sequence for meaningful KV-cache
prompt = "The history of artificial intelligence began in the mid-20th century when researchers first started exploring the possibility of creating machines that could think and learn. Early work focused on symbolic reasoning and expert systems. In the 1980s, connectionist approaches emerged, leading to the development of neural networks. The real breakthrough came in 2012 with deep learning, when convolutional neural networks achieved state-of-the-art results on ImageNet. Since then, the field has exploded with advances in natural language processing, computer vision, and reinforcement learning. Today, large language models with billions of parameters can generate human-like text, solve complex problems, and even write code. The future promises even more remarkable capabilities as models become larger, more efficient, and better aligned with human values."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\nPrompt length: {inputs.input_ids.shape[1]} tokens")

# Forward pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_key_values = outputs.past_key_values
num_layers = len(past_key_values)
seq_len = inputs.input_ids.shape[1]

print(f"Layers: {num_layers}, Sequence length: {seq_len}")

# ---- Test 1: Direct WAL encoding on KV-cache ----
print("\n" + "=" * 60)
print("TEST 1: Direct WAL Encoding")
print("=" * 60)

def encode_decode_kv(kv_tensor, K=256, C=16):
    """Encode and decode a KV tensor with WAL."""
    flat = kv_tensor.float().cpu().reshape(-1)
    atoms = build_l0_atoms(flat, K=K, iters=3)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(flat, atoms, coeffs)
    
    # Decode
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(atoms.numel())]
    atom_table = AtomTableV1(atoms, atom_defs)
    decoded = wal_decode_v1(prog, atom_table, coeffs)
    
    mse = (flat - decoded).pow(2).mean().item()
    rel_mse = mse / flat.pow(2).mean().item()
    return mse, rel_mse, decoded.reshape(kv_tensor.shape)

# Test on layer 40
layer_idx = 40
k_raw = past_key_values[layer_idx][0].float().cpu()  # [1, 8, seq, 128]
v_raw = past_key_values[layer_idx][1].float().cpu()

print(f"\nLayer {layer_idx}:")
print(f"  K shape: {list(k_raw.shape)}")
print(f"  V shape: {list(v_raw.shape)}")

# K with K=256, C=16 (standard weight budget)
print("\n  Encoding K (K=256, C=16)...")
k_mse, k_rel, k_dec = encode_decode_kv(k_raw, K=256, C=16)
print(f"    MSE: {k_mse:.8f}, relMSE: {k_rel:.8f}")

# V with K=256, C=16
print("\n  Encoding V (K=256, C=16)...")
v_mse, v_rel, v_dec = encode_decode_kv(v_raw, K=256, C=16)
print(f"    MSE: {v_mse:.8f}, relMSE: {v_rel:.8f}")

# V with smaller budget (K=64, C=8) - exploit low entropy
print("\n  Encoding V with SMALL budget (K=64, C=8)...")
v_mse_small, v_rel_small, v_dec_small = encode_decode_kv(v_raw, K=64, C=8)
print(f"    MSE: {v_mse_small:.8f}, relMSE: {v_rel_small:.8f}")

# ---- Test 2: Delta encoding ----
print("\n" + "=" * 60)
print("TEST 2: Delta Encoding (exploit temporal correlation)")
print("=" * 60)

def delta_encode_decode(kv_tensor, K=256, C=16):
    """Encode differences between adjacent positions."""
    # Compute deltas: [batch, heads, seq-1, dim]
    deltas = kv_tensor[:, :, 1:, :] - kv_tensor[:, :, :-1, :]
    
    # Flatten deltas
    flat_deltas = deltas.float().cpu().reshape(-1)
    atoms = build_l0_atoms(flat_deltas, K=K, iters=3)
    coeffs = build_coeff_table(flat_deltas, atoms, C=C, iters=3)
    prog, recon_deltas = wal_encode_v1(flat_deltas, atoms, coeffs)
    
    # Decode deltas
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(atoms.numel())]
    atom_table = AtomTableV1(atoms, atom_defs)
    decoded_deltas = wal_decode_v1(prog, atom_table, coeffs)
    decoded_deltas = decoded_deltas.reshape(deltas.shape)
    
    # Reconstruct: first position + cumulative deltas
    reconstructed = torch.zeros_like(kv_tensor)
    reconstructed[:, :, 0, :] = kv_tensor[:, :, 0, :]
    for t in range(1, kv_tensor.shape[2]):
        reconstructed[:, :, t, :] = reconstructed[:, :, t-1, :] + decoded_deltas[:, :, t-1, :]
    
    mse = (kv_tensor - reconstructed).pow(2).mean().item()
    rel_mse = mse / kv_tensor.pow(2).mean().item()
    return mse, rel_mse

print("\n  Delta encoding K (K=256, C=16)...")
dk_mse, dk_rel = delta_encode_decode(k_raw, K=256, C=16)
print(f"    MSE: {dk_mse:.8f}, relMSE: {dk_rel:.8f}")
print(f"    vs direct: relMSE {k_rel:.8f} → {dk_rel:.8f}")

print("\n  Delta encoding V (K=64, C=8)...")
dv_mse, dv_rel = delta_encode_decode(v_raw, K=64, C=8)
print(f"    MSE: {dv_mse:.8f}, relMSE: {dv_rel:.8f}")
print(f"    vs direct: relMSE {v_rel_small:.8f} → {dv_rel:.8f}")

# ---- Test 3: Per-head encoding ----
print("\n" + "=" * 60)
print("TEST 3: Per-Head Atoms vs Global Atoms")
print("=" * 60)

def encode_per_head(kv_tensor, K=64, C=8):
    """Encode each head separately."""
    batch, heads, seq, dim = kv_tensor.shape
    total_mse = 0
    
    for h in range(heads):
        head_tensor = kv_tensor[0, h]  # [seq, dim]
        flat = head_tensor.float().cpu().reshape(-1)
        atoms = build_l0_atoms(flat, K=K, iters=3)
        coeffs = build_coeff_table(flat, atoms, C=C, iters=3)
        prog, recon = wal_encode_v1(flat, atoms, coeffs)
        mse = (flat - recon).pow(2).mean().item()
        total_mse += mse
    
    return total_mse / heads

print("\n  Global atoms (V, K=64, C=8)...")
v_global_mse = encode_decode_kv(v_raw, K=64, C=8)[0]
print(f"    MSE: {v_global_mse:.8f}")

print("\n  Per-head atoms (V, K=64, C=8)...")
v_perhead_mse = encode_per_head(v_raw, K=64, C=8)
print(f"    MSE: {v_perhead_mse:.8f}")
print(f"    Improvement: {(1 - v_perhead_mse/v_global_mse)*100:.1f}%")

# ---- Test 4: Cross-layer budget allocation ----
print("\n" + "=" * 60)
print("TEST 4: Cross-Layer Quality (sample 10 layers)")
print("=" * 60)

sample_layers = [0, 10, 20, 30, 40, 50, 60, 70, 75, 79]
v_results = []

for layer_idx in sample_layers:
    v = past_key_values[layer_idx][1].float().cpu()
    mse, rel, _ = encode_decode_kv(v, K=64, C=8)
    v_results.append((layer_idx, mse, rel))
    print(f"  Layer {layer_idx:2d}: MSE={mse:.8f}, relMSE={rel:.8f}")

# ---- Test 5: Bit-rate sweep for V-cache ----
print("\n" + "=" * 60)
print("TEST 5: Bit-Rate Sweep for V-cache")
print("=" * 60)

v = past_key_values[40][1].float().cpu()
for K in [16, 32, 64, 128, 256]:
    for C in [4, 8, 16]:
        bits = int(np.ceil(np.log2(K))) + int(np.ceil(np.log2(C)))
        mse, rel, _ = encode_decode_kv(v, K=K, C=C)
        print(f"  K={K:3d}, C={C:2d} ({bits:2d} bits): relMSE={rel:.8f}")

# ---- Summary ----
print("\n" + "=" * 60)
print("M85: SUMMARY")
print("=" * 60)
print("\nDirect WAL encoding:")
print(f"  K-cache (K=256,C=16): relMSE={k_rel:.8f}")
print(f"  V-cache (K=256,C=16): relMSE={v_rel:.8f}")
print(f"  V-cache (K=64,C=8):   relMSE={v_rel_small:.8f}")
print("\nDelta encoding:")
print(f"  K-cache: relMSE={dk_rel:.8f} ({'better' if dk_rel < k_rel else 'worse'} than direct)")
print(f"  V-cache: relMSE={dv_rel:.8f} ({'better' if dv_rel < v_rel_small else 'worse'} than direct)")
print("\nPer-head vs global:")
print(f"  Global:  MSE={v_global_mse:.8f}")
print(f"  Per-head: MSE={v_perhead_mse:.8f}")
print(f"  Improvement: {(1 - v_perhead_mse/v_global_mse)*100:.1f}%")

print("\n" + "=" * 60)
print("M85: COMPLETE")
print("=" * 60)
