#!/usr/bin/env python3
"""M45: WAL Scalar Prototype — batched CPU, sample subset for speed."""
import torch
from transformers import AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes
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

target_name = "model.layers.60.mlp.gate_proj.weight"
for name, param in model.named_parameters():
    if name == target_name:
        w_orig = param.data.float().cpu()
        break

print(f"WAL Scalar Prototype on {target_name}")
print(f"Shape: {w_orig.shape}")

# Use subset: first 100K weights for quick prototype
SUBSET = 100_000
w_orig = w_orig.reshape(-1)[:SUBSET]
row_scale = w_orig.abs().amax().unsqueeze(0).clamp_min(1e-8)
w_norm = w_orig / row_scale

print(f"Subset: {SUBSET:,} weights")

# --- DRL v2 Reference (K=128, lmax=8) ---
ladder = torch.tensor([0.5 ** i for i in range(8)], dtype=torch.float32)
enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=8)
cb, ids = build_codebook(enc.digits, enc.stop_depth, 8)
route_values = cb.digits.to(torch.float32) @ ladder
freq = torch.bincount(ids.reshape(-1).long(), minlength=cb.keys.numel()).float()
_, top_idx = freq.topk(128)
centers = route_values[top_idx].clone()

for _ in range(10):
    dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
    assignments = dist.argmin(dim=1)
    new_centers = torch.zeros_like(centers)
    for c in range(128):
        mask = assignments == c
        if mask.any():
            wg = freq[mask]
            new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
    centers = new_centers

# Batched assignment
w_assignments = torch.empty(SUBSET, dtype=torch.int64)
for start in range(0, SUBSET, 50_000):
    end = min(start + 50_000, SUBSET)
    w_assignments[start:end] = (w_norm[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)

w_hat_drl = centers[w_assignments] * row_scale
rel_mse_drl = ((w_hat_drl - w_orig) ** 2).mean() / (w_orig ** 2).mean()
print(f"DRL v2 (K=128, lmax=8): relMSE={rel_mse_drl:.6f}")

# --- WAL Scalar v0.1 ---
K = 128
lmax = 2

# Fast k-means++ on CPU
atoms = torch.zeros(K, dtype=torch.float32)
atoms[0] = w_norm[torch.randint(0, SUBSET, (1,))]
for k in range(1, K):
    dists = (w_norm.unsqueeze(1) - atoms[:k].unsqueeze(0)).abs().min(dim=1)[0]
    probs = dists / dists.sum()
    cumprobs = probs.cumsum(dim=0)
    r = torch.rand(1)
    idx = (cumprobs >= r).nonzero(as_tuple=True)[0][0]
    atoms[k] = w_norm[idx]

# K-means (5 iters, batched)
for it in range(5):
    assignments = torch.empty(SUBSET, dtype=torch.int64)
    for start in range(0, SUBSET, 50_000):
        end = min(start + 50_000, SUBSET)
        assignments[start:end] = (w_norm[start:end].unsqueeze(1) - atoms.unsqueeze(0)).abs().argmin(dim=1)
    for k in range(K):
        mask = assignments == k
        if mask.any():
            atoms[k] = w_norm[mask].mean()

print(f"Atoms: min={atoms.min():.6f}, max={atoms.max():.6f}")

# Greedy program encoding (batched)
programs = torch.zeros(SUBSET, lmax, 2, dtype=torch.int64)
residual = w_norm.clone()

for step in range(lmax):
    best_ids = torch.empty(SUBSET, dtype=torch.int64)
    best_signs = torch.empty(SUBSET, dtype=torch.int64)
    
    for start in range(0, SUBSET, 50_000):
        end = min(start + 50_000, SUBSET)
        batch = residual[start:end]
        scores_pos = (batch.unsqueeze(1) - atoms.unsqueeze(0)) ** 2
        scores_neg = (batch.unsqueeze(1) + atoms.unsqueeze(0)) ** 2
        min_pos, id_pos = scores_pos.min(dim=1)
        min_neg, id_neg = scores_neg.min(dim=1)
        use_pos = min_pos < min_neg
        best_ids[start:end] = torch.where(use_pos, id_pos, id_neg)
        best_signs[start:end] = torch.where(use_pos, torch.tensor(1), torch.tensor(-1))
    
    programs[:, step, 0] = best_ids
    programs[:, step, 1] = best_signs
    residual -= atoms[best_ids] * best_signs.float()
    print(f"  Step {step+1}: residual RMSE={residual.pow(2).mean().sqrt():.6f}")

# Decode
recon = torch.zeros_like(w_norm)
for step in range(lmax):
    recon += atoms[programs[:, step, 0]] * programs[:, step, 1].float()

w_hat_wal = recon * row_scale
rel_mse_wal = ((w_hat_wal - w_orig) ** 2).mean() / (w_orig ** 2).mean()
print(f"\nWAL Scalar (K={K}, lmax={lmax}): relMSE={rel_mse_wal:.6f}")

# Compare
x = torch.randn(SUBSET, 1000)
y_orig = w_orig.unsqueeze(1) * x
y_drl = w_hat_drl.unsqueeze(1) * x
y_wal = w_hat_wal.unsqueeze(1) * x
corr_drl = torch.nn.functional.cosine_similarity(y_orig, y_drl, dim=0).mean()
corr_wal = torch.nn.functional.cosine_similarity(y_orig, y_wal, dim=0).mean()
print(f"Output corr DRL: {corr_drl:.6f}")
print(f"Output corr WAL: {corr_wal:.6f}")

unique_progs = torch.unique(programs.view(-1, 2*lmax), dim=0)
print(f"Unique programs: {unique_progs.shape[0]:,}")
