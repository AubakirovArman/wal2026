#!/usr/bin/env python3
"""M43g: Check dtype and range after encoding."""
import torch
from transformers import AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes, rel_mse
from codebook import build_codebook

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

# Encode one scalar layer
name = "model.layers.10.self_attn.o_proj.weight"
param = dict(model.named_parameters())[name]
w = param.data.float()
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
ladder = torch.tensor([0.5 ** i for i in range(12)], device=w.device, dtype=torch.float32)
enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=8)
cb, ids = build_codebook(enc.digits, enc.stop_depth, 8)
route_values = cb.digits.to(torch.float32) @ ladder[:8]
K = cb.keys.numel()
freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
_, top_idx = freq.topk(min(128, K))
centers = route_values[top_idx].clone()
for _ in range(10):
    dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
    assignments = dist.argmin(dim=1)
    new_centers = torch.zeros_like(centers)
    for c in range(min(128, K)):
        mask = assignments == c
        if mask.any():
            wg = freq[mask]
            new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
    centers = new_centers

w_flat = w_norm.reshape(-1)
w_assignments = torch.empty_like(w_flat, dtype=torch.int64)
for start in range(0, w_flat.numel(), 512*1024):
    end = min(start + 512*1024, w_flat.numel())
    w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)

w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
w_hat = w_hat_norm * row_scale

print(f"param dtype: {param.dtype}")
print(f"param device: {param.device}")
print(f"w_hat dtype: {w_hat.dtype}")
print(f"w_hat device: {w_hat.device}")
print(f"w_hat min: {w_hat.min():.6f}, max: {w_hat.max():.6f}")
print(f"w min: {w.min():.6f}, max: {w.max():.6f}")

# Check if copy_ works correctly
param_copy = param.data.clone()
param.data.copy_(w_hat.to(param.dtype))
print(f"After copy_: min={param.data.min():.6f}, max={param.data.max():.6f}")
print(f"diff max={(param.data - param_copy).abs().max():.6f}")
print(f"NaN after copy: {param.data.isnan().any().item()}")
print(f"Inf after copy: {param.data.isinf().any().item()}")
