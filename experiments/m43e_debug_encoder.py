#!/usr/bin/env python3
"""M43e: Debug scalar encoder on a single weight matrix."""
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

# Pick a smooth layer
name = "model.layers.10.self_attn.o_proj.weight"
param = dict(model.named_parameters())[name]
w = param.data.float()
print(f"Original {name}: shape={tuple(w.shape)}, mean={w.mean():.4f}, std={w.std():.4f}")

row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale
std = w_norm.std().item()
print(f"After row-norm: std={std:.4f}")

ladder = torch.tensor([0.5 ** i for i in range(12)], device=w.device, dtype=torch.float32)
l_max = 8
K_target = 128

enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
print(f"EncodedRoutes: digits shape={enc.digits.shape}, stop_depth max={enc.stop_depth.max().item()}")

cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
print(f"Codebook: keys={cb.keys.numel()}, digits shape={cb.digits.shape}")

route_values = cb.digits.to(torch.float32) @ ladder[:l_max]
print(f"route_values: shape={route_values.shape}, mean={route_values.mean():.4f}, std={route_values.std():.4f}")

K = cb.keys.numel()
freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
_, top_idx = freq.topk(min(K_target, K))
centers = route_values[top_idx].clone()
print(f"centers: shape={centers.shape}, mean={centers.mean():.4f}, std={centers.std():.4f}")

for i in range(5):
    dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
    assignments = dist.argmin(dim=1)
    new_centers = torch.zeros_like(centers)
    for c in range(min(K_target, K)):
        mask = assignments == c
        if mask.any():
            wg = freq[mask]
            new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
    centers = new_centers
    if i % 2 == 0:
        print(f"  iter {i}: centers mean={centers.mean():.4f}, std={centers.std():.4f}, nan={centers.isnan().any().item()}")

w_flat = w_norm.reshape(-1)
w_assignments = torch.empty_like(w_flat, dtype=torch.int64)
batch_size = 512 * 1024
for start in range(0, w_flat.numel(), batch_size):
    end = min(start + batch_size, w_flat.numel())
    w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)

w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
w_hat = w_hat_norm * row_scale

print(f"w_hat: shape={tuple(w_hat.shape)}, mean={w_hat.mean():.4f}, std={w_hat.std():.4f}")
print(f"rel_mse={rel_mse(w, w_hat).item():.6f}")
print(f"max_diff={(w - w_hat).abs().max().item():.4f}")
