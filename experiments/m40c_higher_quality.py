#!/usr/bin/env python3
"""M40c: Higher-quality hybrid encoder (scalar K=128, VRE cb=1024)."""
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes, rel_mse
from codebook import build_codebook

DEVICE = torch.device("cuda:3")


def build_vector_codebook(blocks, cb_size, iters=12):
    N, D = blocks.shape
    blocks = blocks.to(torch.float32)
    codebook = torch.zeros(cb_size, D, device=blocks.device, dtype=torch.float32)
    codebook[0] = blocks[torch.randint(0, N, (1,))]
    for i in range(1, cb_size):
        dists = torch.cdist(blocks, codebook[:i]).min(dim=1)[0]
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1)
        codebook[i] = blocks[idx]
    for _ in range(iters):
        dists = torch.cdist(blocks, codebook)
        ids = dists.argmin(dim=1)
        for k in range(cb_size):
            mask = ids == k
            if mask.any():
                codebook[k] = blocks[mask].mean(dim=0)
    return codebook


def encode_blocks_rvq(blocks, codebook, l_max):
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


def decode_blocks_rvq(digits, ids, codebook, stop_depth, block_size, rows, cols):
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
    return recon.reshape(num_br, num_bc, block_size, block_size).permute(0, 2, 1, 3).reshape(rows, cols)


def encode_scalar_drl(w_norm, ladder, l_max, K_target):
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder
    K = cb.keys.numel()
    freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
    _, top_idx = freq.topk(min(K_target, K))
    centers = route_values[top_idx].clone()
    for _ in range(20):
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
    batch = 512 * 1024
    for start in range(0, w_flat.numel(), batch):
        end = min(start + batch, w_flat.numel())
        w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    return centers[w_assignments].reshape(w_norm.shape)


def hybrid_encode_weight(w, coarse_ladder, vre_cb_size=1024, vre_lmax=10, scalar_K=128, scalar_lmax=8, spiky_threshold=0.08):
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    is_spiky = std < spiky_threshold

    if is_spiky:
        rows, cols = w_norm.shape
        block_size = 4
        if rows % block_size != 0 or cols % block_size != 0:
            pad_r = (block_size - rows % block_size) % block_size
            pad_c = (block_size - cols % block_size) % block_size
            w_pad = torch.nn.functional.pad(w_norm, (0, pad_c, 0, pad_r))
            rows_p, cols_p = w_pad.shape
        else:
            w_pad = w_norm
            rows_p, cols_p = rows, cols
            pad_r = pad_c = 0

        num_br = rows_p // block_size
        num_bc = cols_p // block_size
        blocks = w_pad.view(num_br, block_size, num_bc, block_size).permute(0, 2, 1, 3).reshape(num_br * num_bc, block_size * block_size)
        codebook = build_vector_codebook(blocks, vre_cb_size, iters=12)
        digits, ids, stop_depth = encode_blocks_rvq(blocks, codebook, vre_lmax)
        recon = decode_blocks_rvq(digits, ids, codebook, stop_depth, block_size, rows_p, cols_p)
        if pad_r > 0 or pad_c > 0:
            recon = recon[:rows, :cols]
        w_hat = recon * row_scale
    else:
        ladder = torch.tensor(coarse_ladder, device=w.device, dtype=torch.float32)
        w_hat_norm = encode_scalar_drl(w_norm, ladder, scalar_lmax, scalar_K)
        w_hat = w_hat_norm * row_scale

    return w_hat


def main():
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(DEVICE)

    with torch.no_grad():
        baseline_ppl = math.exp(model(input_ids, labels=input_ids).loss.item())
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    coarse_ladder = [1.0 * (0.5 ** i) for i in range(8)]

    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) == 2:
            if "embed_tokens" in name or "lm_head" in name:
                print(f"Skipping {name}")
                continue
            w = param.data
            w_hat = hybrid_encode_weight(w, coarse_ladder)
            param.data.copy_(w_hat)
            rmse = rel_mse(w, w_hat.to(torch.float16)).item()
            print(f"  {name}: relMSE={rmse:.4f}")

    with torch.no_grad():
        encoded_ppl = math.exp(model(input_ids, labels=input_ids).loss.item())
    print(f"\nEncoded PPL: {encoded_ppl:.4f}")
    print(f"Change: {encoded_ppl - baseline_ppl:+.4f} ({100*(encoded_ppl/baseline_ppl-1):.2f}%)")


if __name__ == "__main__":
    main()
