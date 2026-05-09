"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M43f: Debug VRE encoder on a single spiky weight matrix."""
import torch
from transformers import AutoModelForCausalLM
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import rel_mse

DEVICE = torch.device("cuda:2")
model_name = "unsloth/Llama-3.3-70B-Instruct"
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

name = "model.layers.0.self_attn.q_proj.weight"
param = dict(model.named_parameters())[name]
w = param.data.float()
print(f"Original {name}: shape={tuple(w.shape)}, mean={w.mean():.4f}, std={w.std():.4f}")

row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
std = w_norm.std().item()
print(f"After row-norm: std={std:.4f}")

rows, cols = w_norm.shape
block_size = 4
num_br = rows // block_size
num_bc = cols // block_size
blocks = (
    w_norm.view(num_br, block_size, num_bc, block_size)
    .permute(0, 2, 1, 3)
    .reshape(num_br * num_bc, block_size * block_size)
)
print(f"Blocks: shape={tuple(blocks.shape)}")

# Build codebook (GPU)
blocks_gpu = blocks.to(DEVICE)
cb_size = 512
idx = torch.randperm(blocks_gpu.shape[0], device=blocks_gpu.device)[:cb_size]
codebook = blocks_gpu[idx].clone()
for _ in range(5):
    dists = torch.cdist(blocks_gpu, codebook)
    ids = dists.argmin(dim=1)
    for k in range(cb_size):
        mask = ids == k
        if mask.any():
            codebook[k] = blocks_gpu[mask].mean(dim=0)

# Encode
l_max = 8
N, D = blocks_gpu.shape
K = codebook.shape[0]
digits = torch.zeros(N, l_max, dtype=torch.int8, device="cpu")
ids_cpu = torch.zeros(N, l_max, dtype=torch.int64, device="cpu")
stop_depth = torch.zeros(N, dtype=torch.int32, device="cpu")
residual = blocks_gpu.clone()
cb_norms = (codebook ** 2).sum(dim=1)

batch_size = 262144
for stage in range(l_max):
    any_take = False
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = residual[start:end]
        dots = batch @ codebook.T
        score_pos = (batch ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) - 2 * dots
        score_neg = (batch ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) + 2 * dots
        score_zero = (batch ** 2).sum(dim=1, keepdim=True).expand(-1, K)
        scores = torch.stack([score_neg, score_zero, score_pos], dim=2)
        flat_idx = scores.reshape(end - start, -1).argmin(dim=1)
        best_k = flat_idx // 3
        best_d = (flat_idx % 3) - 1
        zero_scores = (batch ** 2).sum(dim=1)
        min_scores = scores.reshape(end - start, -1).min(dim=1)[0]
        take = (best_d != 0) & (min_scores < zero_scores * 0.999)
        if take.any():
            any_take = True
            digits[start:end][take, stage] = best_d[take].to(torch.int8).cpu()
            ids_cpu[start:end][take, stage] = best_k[take].cpu()
            stop_depth[start:end][take] = stage + 1
            chosen = codebook[best_k[take]] * best_d[take].unsqueeze(1).to(torch.float32)
            residual[start:end][take] -= chosen
    if not any_take:
        break

# Decode
recon = torch.zeros(N, D, dtype=torch.float32)
for stage in range(l_max):
    mask = stop_depth > stage
    if mask.any():
        d = digits[mask, stage].to(torch.float32)
        idx = ids_cpu[mask, stage]
        recon[mask] += codebook[idx].cpu() * d.unsqueeze(1)

w_hat = (
    recon.reshape(num_br, num_bc, block_size, block_size)
    .permute(0, 2, 1, 3)
    .reshape(rows, cols)
) * row_scale

print(f"w_hat: shape={tuple(w_hat.shape)}, mean={w_hat.mean():.4f}, std={w_hat.std():.4f}")
print(f"rel_mse={rel_mse(w, w_hat).item():.6f}")
print(f"max_diff={(w - w_hat).abs().max().item():.4f}")

# Check for NaN/Inf
print(f"NaN in w_hat: {w_hat.isnan().any().item()}")
print(f"Inf in w_hat: {w_hat.isinf().any().item()}")
