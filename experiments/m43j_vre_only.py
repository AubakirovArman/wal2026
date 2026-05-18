#!/usr/bin/env python3
"""M43j: Encode only one spiky layer with VRE, keep everything else original."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path
import math

DEVICE = torch.device("cuda:3")
model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 10

VRE_CB_SIZE = 512
VRE_LMAX = 8


def build_vector_codebook_gpu(blocks, cb_size, iters=5):
    idx = torch.randperm(blocks.shape[0], device=blocks.device)[:cb_size]
    codebook = blocks[idx].clone()
    for _ in range(iters):
        ids = torch.empty(blocks.shape[0], dtype=torch.int64, device=blocks.device)
        batch_size = 262144
        for start in range(0, blocks.shape[0], batch_size):
            end = min(start + batch_size, blocks.shape[0])
            dists = torch.cdist(blocks[start:end], codebook)
            ids[start:end] = dists.argmin(dim=1)
        for k in range(cb_size):
            mask = ids == k
            if mask.any():
                codebook[k] = blocks[mask].mean(dim=0)
    return codebook


def encode_blocks_rvq_gpu_batched(blocks, codebook, l_max, batch_size=262144):
    N, D = blocks.shape
    K = codebook.shape[0]
    digits = torch.zeros(N, l_max, dtype=torch.int8, device="cpu")
    ids_cpu = torch.zeros(N, l_max, dtype=torch.int64, device="cpu")
    stop_depth = torch.zeros(N, dtype=torch.int32, device="cpu")
    residual = blocks.clone()
    cb_norms = (codebook ** 2).sum(dim=1)

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
    return digits, ids_cpu, stop_depth


def decode_blocks_rvq_batched(digits, ids, codebook, stop_depth, block_size, rows, cols):
    N, l_max = digits.shape
    D = codebook.shape[1]
    recon = torch.zeros(N, D, dtype=torch.float32)
    for stage in range(l_max):
        mask = stop_depth > stage
        if mask.any():
            d = digits[mask, stage].to(torch.float32)
            idx = ids[mask, stage]
            recon[mask] += codebook[idx].cpu() * d.unsqueeze(1)
    num_br = rows // block_size
    num_bc = cols // block_size
    return (
        recon.reshape(num_br, num_bc, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )


print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Encode only layer 0 q_proj
name = "model.language_model.layers.0.self_attn.q_proj.weight"
param = dict(model.named_parameters())[name]
w = param.data.float()
row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
w_norm = w / row_scale

rows, cols = w_norm.shape
block_size = 4
num_br = rows // block_size
num_bc = cols // block_size
blocks = (
    w_norm.view(num_br, block_size, num_bc, block_size)
    .permute(0, 2, 1, 3)
    .reshape(num_br * num_bc, block_size * block_size)
)
blocks_gpu = blocks.to(DEVICE)
codebook = build_vector_codebook_gpu(blocks_gpu, VRE_CB_SIZE, iters=5)
digits, ids_cpu, stop_depth = encode_blocks_rvq_gpu_batched(blocks_gpu, codebook, VRE_LMAX)
recon = decode_blocks_rvq_batched(digits, ids_cpu, codebook, stop_depth, block_size, rows, cols)
w_hat = recon.to(param.device) * row_scale

print(f"Original: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}, std={w.std():.4f}")
print(f"Encoded:  min={w_hat.min():.4f}, max={w_hat.max():.4f}, mean={w_hat.mean():.4f}, std={w_hat.std():.4f}")
print(f"rel_mse={((w - w_hat)**2).mean() / (w**2).mean():.6f}")

param.data.copy_(w_hat.to(param.dtype))

# PPL
print("Loading WikiText-2...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
num_steps = 0
for begin_loc in range(0, seq_len, stride):
    if num_steps >= max_samples:
        break
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"\nVRE single-layer PPL (first {end_loc} tokens): {ppl.item():.2f}")
