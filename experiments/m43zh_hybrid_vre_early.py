#!/usr/bin/env python3
"""M43zh: VRE for std<0.03, scalar K=128 for smooth, skip 0.03<=std<0.08."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path

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
VRE_THRESHOLD = 0.03
SPIKY_THRESHOLD = 0.08
COARSE_LADDER = [0.5 ** i for i in range(12)]


def encode_scalar_drl_batched(w_norm, ladder, l_max, K_target):
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder[:l_max]
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
    w_assignments = torch.empty_like(w_flat, dtype=torch.int64)
    batch_size = 512 * 1024
    for start in range(0, w_flat.numel(), batch_size):
        end = min(start + batch_size, w_flat.numel())
        w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
    return w_hat_norm


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

print("Applying adaptive hybrid encoder...")
stats = {"vre": 0, "scalar": 0, "skipped": 0}
for name, param in list(model.named_parameters()):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue

    w = param.data.float()
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()

    if std < VRE_THRESHOLD:
        # VRE path
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
        param.data.copy_(w_hat.to(param.dtype))
        stats["vre"] += 1
    elif std < SPIKY_THRESHOLD:
        # Mid-spiky: skip (keep original)
        stats["skipped"] += 1
    else:
        # Smooth: scalar
        ladder = torch.tensor(COARSE_LADDER, device=w.device, dtype=torch.float32)
        w_hat_norm = encode_scalar_drl_batched(w_norm, ladder, SCALAR_LMAX, SCALAR_K)
        w_hat = w_hat_norm * row_scale
        param.data.copy_(w_hat.to(param.dtype))
        stats["scalar"] += 1

    total_enc = stats["vre"] + stats["scalar"]
    if total_enc % 50 == 0 or total_enc <= 10:
        print(f"  Encoded {total_enc} params (VRE={stats['vre']}, scalar={stats['scalar']}, skipped={stats['skipped']})")

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
print(f"\nAdaptive hybrid PPL (first {end_loc} tokens): {ppl.item():.2f}")
