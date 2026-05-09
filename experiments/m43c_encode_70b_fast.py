#!/usr/bin/env python3
"""M43c: Fast hybrid encoder for 70B using GPU acceleration."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes, rel_mse
from codebook import build_codebook

DEVICE = torch.device("cuda:2")
model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 10

VRE_CB_SIZE = 512
VRE_LMAX = 8
SCALAR_K = 128
SCALAR_LMAX = 8
SPIKY_THRESHOLD = 0.08
COARSE_LADDER = [0.5 ** i for i in range(12)]


def build_vector_codebook_gpu(blocks, cb_size, iters=5):
    N, D = blocks.shape
    codebook = torch.zeros(cb_size, D, device=blocks.device, dtype=torch.float32)
    idx = torch.randperm(N, device=blocks.device)[:cb_size]
    codebook = blocks[idx].clone()
    for _ in range(iters):
        dists = torch.cdist(blocks, codebook)
        ids = dists.argmin(dim=1)
        for k in range(cb_size):
            mask = ids == k
            if mask.any():
                codebook[k] = blocks[mask].mean(dim=0)
    return codebook


def encode_blocks_rvq_gpu(blocks, codebook, l_max):
    N, D = blocks.shape
    K = codebook.shape[0]
    digits = torch.zeros(N, l_max, dtype=torch.int8, device=blocks.device)
    ids = torch.zeros(N, l_max, dtype=torch.int64, device=blocks.device)
    stop_depth = torch.zeros(N, dtype=torch.int32, device=blocks.device)
    residual = blocks.clone()
    cb_norms = (codebook ** 2).sum(dim=1)
    for stage in range(l_max):
        dots = residual @ codebook.T
        score_pos = (residual ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) - 2 * dots
        score_neg = (residual ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) + 2 * dots
        score_zero = (residual ** 2).sum(dim=1, keepdim=True).expand(-1, K)
        scores = torch.stack([score_neg, score_zero, score_pos], dim=2)
        flat_idx = scores.reshape(N, -1).argmin(dim=1)
        best_k = flat_idx // 3
        best_d = (flat_idx % 3) - 1
        zero_scores = (residual ** 2).sum(dim=1)
        min_scores = scores.reshape(N, -1).min(dim=1)[0]
        take = (best_d != 0) & (min_scores < zero_scores * 0.999)
        if not take.any():
            break
        digits[take, stage] = best_d[take].to(torch.int8)
        ids[take, stage] = best_k[take]
        stop_depth[take] = stage + 1
        chosen = codebook[best_k[take]] * best_d[take].unsqueeze(1).to(torch.float32)
        residual[take] -= chosen
    return digits, ids, stop_depth


def decode_blocks_rvq_gpu(digits, ids, codebook, stop_depth, block_size, rows, cols):
    N, l_max = digits.shape
    D = codebook.shape[1]
    recon = torch.zeros(N, D, device=digits.device, dtype=torch.float32)
    for stage in range(l_max):
        mask = stop_depth > stage
        if mask.any():
            d = digits[mask, stage].to(torch.float32)
            idx = ids[mask, stage]
            recon[mask] += codebook[idx] * d.unsqueeze(1)
    num_br = rows // block_size
    num_bc = cols // block_size
    return (
        recon.reshape(num_br, num_bc, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )


def encode_scalar_drl_gpu(w_norm, ladder, l_max, K_target):
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder
    K = cb.keys.numel()
    freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
    _, top_idx = freq.topk(min(K_target, K))
    centers = route_values[top_idx].clone()
    for _ in range(10):
        dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
        assignments = dist.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for c in range(min(K_target, K)):
            mask = assignments == c
            if mask.any():
                wg = freq[mask]
                new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
        centers = new_centers
    w_flat = w_norm.reshape(-1)
    w_assignments = (w_flat.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
    return w_hat_norm


def hybrid_encode_fast(w, coarse_ladder, vre_cb_size=512, vre_lmax=8, scalar_K=128, scalar_lmax=8, spiky_threshold=0.08):
    device = w.device
    w_f = w.float()
    row_scale = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w_f / row_scale
    std = w_norm.std().item()
    is_spiky = std < spiky_threshold

    if is_spiky:
        rows, cols = w_norm.shape
        block_size = 4
        num_br = rows // block_size
        num_bc = cols // block_size
        blocks = (
            w_norm.view(num_br, block_size, num_bc, block_size)
            .permute(0, 2, 1, 3)
            .reshape(num_br * num_bc, block_size * block_size)
        ).to(DEVICE)
        codebook = build_vector_codebook_gpu(blocks, vre_cb_size, iters=5)
        digits, ids, stop_depth = encode_blocks_rvq_gpu(blocks, codebook, vre_lmax)
        recon = decode_blocks_rvq_gpu(digits, ids, codebook, stop_depth, block_size, rows, cols)
        w_hat = recon.to(device) * row_scale

        prog_keys = torch.zeros(digits.shape[0], dtype=torch.int64, device=digits.device)
        for s in range(vre_lmax):
            prog_keys = prog_keys * (vre_cb_size * 3 + 1) + (ids[:, s] * 3 + digits[:, s].long() + 1)
        unique_programs = torch.unique(prog_keys).numel()
        avg_depth = stop_depth.float().mean().item()
        bps = (math.log2(vre_cb_size) * avg_depth + math.log2(vre_lmax)) / (block_size * block_size)

        return {
            "method": "VRE",
            "is_spiky": True,
            "std": std,
            "w_hat": w_hat.to(w.dtype),
            "rel_mse": rel_mse(w, w_hat).item(),
            "bps": bps,
            "unique_programs": int(unique_programs),
            "total_blocks": int(digits.shape[0]),
            "avg_depth": avg_depth,
        }
    else:
        ladder = torch.tensor(coarse_ladder, device=DEVICE, dtype=torch.float32)
        w_norm_gpu = w_norm.to(DEVICE)
        w_hat_norm = encode_scalar_drl_gpu(w_norm_gpu, ladder, scalar_lmax, scalar_K)
        w_hat = w_hat_norm.to(device) * row_scale
        return {
            "method": "scalar_DRL",
            "is_spiky": False,
            "std": std,
            "w_hat": w_hat.to(w.dtype),
            "rel_mse": rel_mse(w, w_hat).item(),
            "bps": math.log2(scalar_K),
        }


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

print("Applying hybrid encoder (GPU-accelerated)...")
stats = {"vre": 0, "scalar": 0, "skipped": 0}
for name, param in list(model.named_parameters()):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue

    result = hybrid_encode_fast(
        param.data,
        coarse_ladder=COARSE_LADDER,
        vre_cb_size=VRE_CB_SIZE,
        vre_lmax=VRE_LMAX,
        scalar_K=SCALAR_K,
        scalar_lmax=SCALAR_LMAX,
        spiky_threshold=SPIKY_THRESHOLD,
    )
    w_hat = result["w_hat"]
    param.data.copy_(w_hat)
    is_spiky = result["is_spiky"]
    del w_hat, result
    torch.cuda.empty_cache()

    if is_spiky:
        stats["vre"] += 1
    else:
        stats["scalar"] += 1

    total_enc = stats["vre"] + stats["scalar"]
    if total_enc % 50 == 0 or total_enc <= 10:
        print(f"  Encoded {total_enc} params (VRE={stats['vre']}, scalar={stats['scalar']})")

print(f"Encoding done: {stats}")

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
print(f"\nEncoded PPL (first {end_loc} tokens): {ppl.item():.2f}")
