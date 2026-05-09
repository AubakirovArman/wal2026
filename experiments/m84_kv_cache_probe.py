#!/usr/bin/env python3
"""M84: KV-cache WAL — Probe.

Understand KV-cache structure from Llama 70B and compare to weight structure.
KV-cache is fundamentally different from weights:
- Dynamic (changes every token)
- Context-dependent
- Temporal structure (earlier tokens influence later)
- Different distribution (activations vs trained weights)

Hypothesis: KV-cache may compress BETTER than weights because:
1. KV values are "smoothed" by attention (time-averaged)
2. Temporal correlation across sequence positions
3. Per-head structure may be more regular than per-layer weights
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("M84: KV-cache WAL Probe")
print("=" * 60)

# ---- Setup ----
# Use first visible GPU (with CUDA_VISIBLE_DEVICES set externally)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (visible GPUs: {torch.cuda.device_count()})")

# Load model
print("\nLoading Llama 3.3 70B...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct")

print(f"Model loaded: {model.config.num_hidden_layers} layers")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num heads: {model.config.num_attention_heads}")
print(f"Head dim: {model.config.hidden_size // model.config.num_attention_heads}")
print(f"Num KV heads: {model.config.num_key_value_heads}")

# Generate a sequence to populate KV-cache
prompt = "The history of artificial intelligence began in"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: '{prompt}'")
print(f"Input tokens: {inputs.input_ids.shape[1]}")

# Forward pass to populate KV-cache
print("\nRunning forward pass to populate KV-cache...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_key_values = outputs.past_key_values
num_layers = len(past_key_values)

print(f"KV-cache layers: {num_layers}")
print(f"KV-cache entries per layer: {len(past_key_values[0])}")  # key, value

# Extract KV-cache statistics
print("\n" + "=" * 60)
print("KV-CACHE STATISTICS")
print("=" * 60)

all_k_flat = []
all_v_flat = []

for layer_idx, (k, v) in enumerate(past_key_values):
    # k, v shapes: [batch, num_kv_heads, seq_len, head_dim]
    k_flat = k.float().cpu().reshape(-1)
    v_flat = v.float().cpu().reshape(-1)
    all_k_flat.append(k_flat)
    all_v_flat.append(v_flat)
    
    if layer_idx < 3 or layer_idx >= num_layers - 3:
        print(f"\nLayer {layer_idx}:")
        print(f"  K shape: {list(k.shape)}, range: [{k_flat.min():.4f}, {k_flat.max():.4f}], mean: {k_flat.mean():.4f}, std: {k_flat.std():.4f}")
        print(f"  V shape: {list(v.shape)}, range: [{v_flat.min():.4f}, {v_flat.max():.4f}], mean: {v_flat.mean():.4f}, std: {v_flat.std():.4f}")
    elif layer_idx == 3:
        print(f"\n  ... ({num_layers - 6} layers omitted) ...")

# Combine all layers
all_k = torch.cat(all_k_flat)
all_v = torch.cat(all_v_flat)

print(f"\n{'='*60}")
print("AGGREGATE STATISTICS (all layers)")
print(f"{'='*60}")
print(f"Total K elements: {all_k.numel():,}")
print(f"Total V elements: {all_v.numel():,}")
print(f"\nKeys:   min={all_k.min():.4f}, max={all_k.max():.4f}, mean={all_k.mean():.4f}, std={all_k.std():.4f}")
print(f"Values: min={all_v.min():.4f}, max={all_v.max():.4f}, mean={all_v.mean():.4f}, std={all_v.std():.4f}")

# Compare to a typical weight matrix
sample_weight = model.model.layers[40].self_attn.q_proj.weight.data.float().cpu().reshape(-1)
print(f"\nWeight (layer 40 q_proj): min={sample_weight.min():.4f}, max={sample_weight.max():.4f}, mean={sample_weight.mean():.4f}, std={sample_weight.std():.4f}")

# ---- Per-head analysis ----
print(f"\n{'='*60}")
print("PER-HEAD ANALYSIS (Layer 40)")
print(f"{'='*60}")

layer_40_k = past_key_values[40][0]  # [batch, num_kv_heads, seq_len, head_dim]
layer_40_v = past_key_values[40][1]

for head_idx in range(min(4, layer_40_k.shape[1])):
    k_head = layer_40_k[0, head_idx].float().cpu().reshape(-1)
    v_head = layer_40_v[0, head_idx].float().cpu().reshape(-1)
    print(f"\nHead {head_idx}:")
    print(f"  K: std={k_head.std():.4f}, abs_mean={k_head.abs().mean():.4f}")
    print(f"  V: std={v_head.std():.4f}, abs_mean={v_head.abs().mean():.4f}")

# ---- Temporal correlation ----
print(f"\n{'='*60}")
print("TEMPORAL CORRELATION (Layer 40, Head 0)")
print(f"{'='*60}")

k_head0 = layer_40_k[0, 0].float().cpu()  # [seq_len, head_dim]
v_head0 = layer_40_v[0, 0].float().cpu()  # [seq_len, head_dim]

# Correlation between adjacent positions
def adj_correlation(t):
    """Average correlation between adjacent sequence positions."""
    if t.shape[0] < 2:
        return 0.0
    corrs = []
    for i in range(t.shape[0] - 1):
        a, b = t[i], t[i+1]
        if a.std() > 0 and b.std() > 0:
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            corrs.append(corr)
    return np.mean(corrs) if corrs else 0.0

k_corr = adj_correlation(k_head0)
v_corr = adj_correlation(v_head0)
print(f"K adjacent-pos correlation: {k_corr:.4f}")
print(f"V adjacent-pos correlation: {v_corr:.4f}")

# Self-correlation across head_dim (how correlated are dimensions within a position?)
k_dim_corr = torch.corrcoef(k_head0.T).abs().mean().item() - 1.0 / k_head0.shape[1]
v_dim_corr = torch.corrcoef(v_head0.T).abs().mean().item() - 1.0 / v_head0.shape[1]
print(f"K dim self-correlation: {k_dim_corr:.4f}")
print(f"V dim self-correlation: {v_dim_corr:.4f}")

# ---- Entropy analysis ----
print(f"\n{'='*60}")
print("ENTROPY ANALYSIS")
print(f"{'='*60}")

# How many unique values?
k_unique = torch.unique(all_k).numel()
v_unique = torch.unique(all_v).numel()
print(f"K unique values: {k_unique:,} / {all_k.numel():,} ({100*k_unique/all_k.numel():.4f}%)")
print(f"V unique values: {v_unique:,} / {all_v.numel():,} ({100*v_unique/all_v.numel():.4f}%)")

# Histogram entropy
def histogram_entropy(x, bins=256):
    hist = torch.histc(x, bins=bins, min=x.min().item(), max=x.max().item())
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -(hist * torch.log2(hist)).sum().item()

k_entropy = histogram_entropy(all_k)
v_entropy = histogram_entropy(all_v)
w_entropy = histogram_entropy(sample_weight)
print(f"\nK value entropy: {k_entropy:.2f} bits (max 8 for 256 bins)")
print(f"V value entropy: {v_entropy:.2f} bits")
print(f"Weight entropy:  {w_entropy:.2f} bits")

# ---- Sparsity ----
print(f"\n{'='*60}")
print("SPARSITY")
print(f"{'='*60}")

k_near_zero = (all_k.abs() < 0.01).float().mean().item()
v_near_zero = (all_v.abs() < 0.01).float().mean().item()
w_near_zero = (sample_weight.abs() < 0.01).float().mean().item()
print(f"K |x| < 0.01: {k_near_zero*100:.2f}%")
print(f"V |x| < 0.01: {v_near_zero*100:.2f}%")
print(f"W |x| < 0.01: {w_near_zero*100:.2f}%")

# ---- Hypothesis test ----
print(f"\n{'='*60}")
print("HYPOTHESIS EVALUATION")
print(f"{'='*60}")

print("H1: KV values are 'smoother' than weights")
print(f"  K std: {all_k.std():.4f} vs Weight std: {sample_weight.std():.4f}")
print(f"  V std: {all_v.std():.4f} vs Weight std: {sample_weight.std():.4f}")
k_smoother = all_k.std() < sample_weight.std()
v_smoother = all_v.std() < sample_weight.std()
print(f"  Result: K={'YES' if k_smoother else 'NO'}, V={'YES' if v_smoother else 'NO'}")

print("\nH2: KV has temporal correlation")
print(f"  K adjacent correlation: {k_corr:.4f}")
print(f"  V adjacent correlation: {v_corr:.4f}")
print(f"  Result: K={'YES' if abs(k_corr) > 0.1 else 'NO'}, V={'YES' if abs(v_corr) > 0.1 else 'NO'}")

print("\nH3: KV has lower entropy than weights")
print(f"  K entropy: {k_entropy:.2f} vs Weight: {w_entropy:.2f}")
print(f"  V entropy: {v_entropy:.2f} vs Weight: {w_entropy:.2f}")
k_lower_ent = k_entropy < w_entropy
v_lower_ent = v_entropy < w_entropy
print(f"  Result: K={'YES' if k_lower_ent else 'NO'}, V={'YES' if v_lower_ent else 'NO'}")

# ---- Summary ----
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total KV elements: {(all_k.numel() + all_v.numel()):,}")
print(f"At bf16: {(all_k.numel() + all_v.numel()) * 2 / 1024**2:.1f} MB")
print(f"At 12-bit WAL: {(all_k.numel() + all_v.numel()) * 12 / 8 / 1024**2:.1f} MB")
print(f"Potential savings: {100 - 100*12/16:.1f}%")

print("\n" + "=" * 60)
print("M84: PROBE COMPLETE")
print("=" * 60)
