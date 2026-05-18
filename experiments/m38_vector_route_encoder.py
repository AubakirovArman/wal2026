#!/usr/bin/env python3
"""M38: Vector Route Encoder (VRE).

Encodes BxB blocks of weights as residual sums over a shared vector codebook.
Each block's "program" is a sequence of (digit, vector_index) choices.

Key differences from scalar DRL v2:
- Operates on vectors (flattened BxB blocks) instead of scalars
- Shared codebook across all blocks creates natural reuse
- Stop-depth per block creates variable-length programs
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from block_vq import BlockRVQEncoding
from route_encoder import rel_mse

DEVICE = torch.device("cuda:3")


def dict_to_bre(d: dict) -> BlockRVQEncoding:
    return BlockRVQEncoding(
        stage_ids=tuple(d["stage_ids"]),
        codebooks=tuple(d["codebooks"]),
        stage_value_dims=tuple(d["stage_value_dims"]),
        stages_per_split=tuple(d["stages_per_split"]),
        stage_scales=d["stage_scales"],
        residual_correction=d["residual_correction"],
        residual_signs=d["residual_signs"],
        residual_scale=d["residual_scale"],
        row_scale=d["row_scale"],
        block_scale=d["block_scale"],
        transform_kind=d["transform_kind"],
        transform_matrix=d["transform_matrix"],
        transform_bias=d["transform_bias"],
        product_splits=d["product_splits"],
        original_shape=tuple(d["original_shape"]),
        padded_cols=d["padded_cols"],
        block_size=d["block_size"],
        sample_rel_mse=d["sample_rel_mse"],
    )


def build_vector_codebook(blocks: torch.Tensor, codebook_size: int, iters: int = 10) -> torch.Tensor:
    """K-means on flattened blocks to build shared vector codebook.

    Args:
        blocks: [num_blocks, block_dim] flattened block vectors
        codebook_size: number of vectors in codebook
        iters: k-means iterations

    Returns:
        codebook: [codebook_size, block_dim]
    """
    N, D = blocks.shape
    blocks = blocks.to(torch.float32)
    # K-means++ initialization
    codebook = torch.zeros(codebook_size, D, device=blocks.device, dtype=torch.float32)
    codebook[0] = blocks[torch.randint(0, N, (1,))]
    for i in range(1, codebook_size):
        dists = torch.cdist(blocks, codebook[:i]).min(dim=1)[0]
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1)
        codebook[i] = blocks[idx]

    for _ in range(iters):
        # Assign
        dists = torch.cdist(blocks, codebook)  # [N, K]
        ids = dists.argmin(dim=1)
        # Update
        for k in range(codebook_size):
            mask = ids == k
            if mask.any():
                codebook[k] = blocks[mask].mean(dim=0)
    return codebook


def encode_blocks_rvq(
    blocks: torch.Tensor,
    codebook: torch.Tensor,
    l_max: int,
    stop_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy residual vector quantization for blocks.

    Args:
        blocks: [num_blocks, block_dim]
        codebook: [codebook_size, block_dim]
        l_max: max number of stages
        stop_threshold: stop if residual norm < threshold

    Returns:
        digits: [num_blocks, l_max] int8
        ids: [num_blocks, l_max] int64 (codebook indices)
        stop_depth: [num_blocks] int32
    """
    N, D = blocks.shape
    K = codebook.shape[0]
    digits = torch.zeros(N, l_max, dtype=torch.int8, device=blocks.device)
    ids = torch.zeros(N, l_max, dtype=torch.int64, device=blocks.device)
    stop_depth = torch.zeros(N, dtype=torch.int32, device=blocks.device)
    residual = blocks.clone()

    # Precompute codebook norms for efficiency
    cb_norms = (codebook ** 2).sum(dim=1)  # [K]

    for stage in range(l_max):
        # For each block, find best codebook vector
        # score = ||residual - c||^2 = ||residual||^2 + ||c||^2 - 2*residual·c
        # Minimizing score equivalent to maximizing residual·c - 0.5*||c||^2
        # But we also allow digit = -1, 0, +1
        # For digit = +1: score = ||residual - c||^2
        # For digit = -1: score = ||residual + c||^2
        # For digit = 0: score = ||residual||^2

        # Compute dot products: [N, K]
        dots = residual @ codebook.T  # [N, K]

        # Scores for +1 and -1
        score_pos = (residual ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) - 2 * dots
        score_neg = (residual ** 2).sum(dim=1, keepdim=True) + cb_norms.unsqueeze(0) + 2 * dots
        score_zero = (residual ** 2).sum(dim=1, keepdim=True).expand(-1, K)

        # Stack: [N, K, 3] for digits (-1, 0, +1)
        scores = torch.stack([score_neg, score_zero, score_pos], dim=2)  # [N, K, 3]

        # Find best (k, digit)
        flat_idx = scores.reshape(N, -1).argmin(dim=1)
        best_k = flat_idx // 3
        best_d = (flat_idx % 3) - 1  # 0->-1, 1->0, 2->+1

        # Only take blocks where best_d != 0 and there is actual improvement
        zero_scores = (residual ** 2).sum(dim=1)
        min_scores = scores.reshape(N, -1).min(dim=1)[0]
        take = (best_d != 0) & (min_scores < zero_scores * 0.999)
        if not take.any():
            break

        digits[take, stage] = best_d[take].to(torch.int8)
        ids[take, stage] = best_k[take]
        stop_depth[take] = stage + 1

        # Update residual
        chosen = codebook[best_k[take]] * best_d[take].unsqueeze(1).to(torch.float32)
        residual[take] -= chosen

        # Check stop threshold
        if stop_threshold > 0:
            res_norms = (residual ** 2).sum(dim=1).sqrt()
            stopped = res_norms < stop_threshold
            if stopped.any():
                # Mark remaining as stopped
                pass  # They'll naturally have digit=0 for future stages

    return digits, ids, stop_depth


def decode_blocks_rvq(digits, ids, codebook, stop_depth) -> torch.Tensor:
    """Decode block programs back to vectors."""
    N, l_max = digits.shape
    D = codebook.shape[1]
    recon = torch.zeros(N, D, device=digits.device, dtype=torch.float32)
    for stage in range(l_max):
        mask = stop_depth > stage
        if mask.any():
            d = digits[mask, stage].to(torch.float32)
            idx = ids[mask, stage]
            recon[mask] += codebook[idx] * d.unsqueeze(1)
    return recon


def run_experiment(enc_path, key, block_size, cb_size, l_max):
    enc = torch.load(enc_path, map_location="cpu")
    g = enc["encodings"][key]["groups"][0]
    bre = dict_to_bre(g)
    w = bre.reconstruct(out_dtype=torch.float32).to(DEVICE)

    # Row normalize
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale

    rows, cols = w_norm.shape
    num_br = rows // block_size
    num_bc = cols // block_size

    # Extract blocks
    blocks = (
        w_norm.view(num_br, block_size, num_bc, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_br * num_bc, block_size * block_size)
    )

    print(f"\n{key}: shape={tuple(w.shape)}, blocks={blocks.shape}")

    # Build codebook
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    codebook = build_vector_codebook(blocks, cb_size, iters=8)
    t1.record()
    torch.cuda.synchronize()
    print(f"  Codebook built: {cb_size} vectors, time={t0.elapsed_time(t1)/1000:.2f}s")

    # Encode
    t0.record()
    digits, ids, stop_depth = encode_blocks_rvq(blocks, codebook, l_max)
    t1.record()
    torch.cuda.synchronize()
    encode_time = t0.elapsed_time(t1) / 1000

    # Decode
    recon_blocks = decode_blocks_rvq(digits, ids, codebook, stop_depth)
    recon = (
        recon_blocks.reshape(num_br, num_bc, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )
    w_hat = recon * row_scale
    rmse = rel_mse(w, w_hat).item()

    # Stats
    unique_programs = 0
    # Pack programs into unique keys
    prog_keys = torch.zeros(digits.shape[0], dtype=torch.int64, device=digits.device)
    for s in range(l_max):
        prog_keys = prog_keys * (cb_size * 3 + 1) + (ids[:, s] * 3 + digits[:, s].long() + 1)
    unique_programs = torch.unique(prog_keys).numel()

    avg_depth = stop_depth.float().mean().item()
    bps = (math.log2(cb_size) * avg_depth + math.log2(l_max)) / (block_size * block_size)

    print(f"  Encode time: {encode_time:.2f}s")
    print(f"  relMSE: {rmse:.6f}")
    print(f"  Unique programs: {unique_programs} / {digits.shape[0]}")
    print(f"  Avg depth: {avg_depth:.2f} / {l_max}")
    print(f"  Approx bps: {bps:.2f}")

    return {
        "layer": key,
        "shape": list(w.shape),
        "block_size": block_size,
        "codebook_size": cb_size,
        "l_max": l_max,
        "br_rmse": bre.sample_rel_mse,
        "rel_mse": rmse,
        "unique_programs": int(unique_programs),
        "total_blocks": int(digits.shape[0]),
        "avg_depth": avg_depth,
        "bps": bps,
        "encode_time_s": encode_time,
    }


def main():
    configs = [
        # (enc_path, key, block_size, cb_size, l_max)
        ("results/m25_l54_q_gu_encodings.pt", "model.language_model.layers.54.self_attn.q_proj", 4, 256, 6),
        ("results/m25_l54_q_gu_encodings.pt", "model.language_model.layers.54.self_attn.q_proj", 4, 64, 6),
        ("results/m25_l54_q_gu_encodings.pt", "model.language_model.layers.54.self_attn.q_proj", 8, 256, 6),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.language_model.layers.0.self_attn.q_proj", 4, 256, 6),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.language_model.layers.0.self_attn.q_proj", 4, 64, 6),
    ]

    all_results = []
    for enc_path, key, bs, cb, lm in configs:
        r = run_experiment(enc_path, key, bs, cb, lm)
        all_results.append(r)

    with open("results/m38_vector_route_encoder.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/m38_vector_route_encoder.json")


if __name__ == "__main__":
    main()
