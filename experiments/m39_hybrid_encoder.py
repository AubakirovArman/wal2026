"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M39: Hybrid encoder — auto-selects VRE (early/spiky) or scalar DRL v2 (late/smooth).

Spiky vs smooth heuristic: std of row-normalized weights.
- std < 0.08  → spiky → VRE (block 4×4, cb=256, lmax=8)
- std >= 0.08 → smooth → scalar DRL v2 M35 (K=16, coarse ladder)

Both methods use row normalization.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from block_vq import BlockRVQEncoding
from codebook import build_codebook
from route_encoder import decode_routes, encode_routes, rel_mse

DEVICE = torch.device("cuda:2")


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


# ── VRE components ──────────────────────────────────────────────────────────

def build_vector_codebook(blocks: torch.Tensor, cb_size: int, iters: int = 12) -> torch.Tensor:
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
    return (
        recon.reshape(num_br, num_bc, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )


# ── Scalar DRL v2 M35 ───────────────────────────────────────────────────────

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
    w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
    return w_hat_norm


# ── Hybrid encoder ──────────────────────────────────────────────────────────

def hybrid_encode(w, coarse_ladder, vre_cb_size=512, vre_lmax=10, scalar_K=16, scalar_lmax=8, spiky_threshold=0.08):
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale
    std = w_norm.std().item()
    is_spiky = std < spiky_threshold

    if is_spiky:
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
        codebook = build_vector_codebook(blocks, vre_cb_size, iters=12)
        digits, ids, stop_depth = encode_blocks_rvq(blocks, codebook, vre_lmax)
        recon = decode_blocks_rvq(digits, ids, codebook, stop_depth, block_size, rows, cols)
        w_hat = recon * row_scale

        # Stats
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
            "rel_mse": rel_mse(w, w_hat).item(),
            "bps": bps,
            "unique_programs": int(unique_programs),
            "total_blocks": int(digits.shape[0]),
            "avg_depth": avg_depth,
            "w_hat": w_hat,
        }
    else:
        # Scalar DRL v2 M35 path
        ladder = torch.tensor(coarse_ladder, device=w.device, dtype=torch.float32)
        w_hat_norm = encode_scalar_drl(w_norm, ladder, scalar_lmax, scalar_K)
        w_hat = w_hat_norm * row_scale
        return {
            "method": "scalar_DRL",
            "is_spiky": False,
            "std": std,
            "rel_mse": rel_mse(w, w_hat).item(),
            "bps": math.log2(scalar_K),
            "w_hat": w_hat,
        }


def main():
    coarse_ladder = [1.0 * (0.5 ** i) for i in range(8)]

    configs = [
        ("results/m25_l54_q_gu_encodings.pt", "model.layers.54.self_attn.q_proj"),
        ("results/m25_l54_q_gu_encodings.pt", "model.layers.54.mlp.gate_proj"),
        ("results/m25_l54_q_gu_encodings.pt", "model.layers.54.mlp.up_proj"),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.layers.0.self_attn.q_proj"),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.layers.0.self_attn.k_proj"),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.layers.0.self_attn.v_proj"),
    ]

    all_results = []
    for enc_path, key in configs:
        print(f"\n=== {key} ===")
        enc = torch.load(enc_path, map_location="cpu")
        g = enc["encodings"][key]["groups"][0]
        bre = dict_to_bre(g)
        w = bre.reconstruct(out_dtype=torch.float32).to(DEVICE)

        r = hybrid_encode(w, coarse_ladder)
        r["layer"] = key
        r["shape"] = list(w.shape)
        r["br_rmse"] = bre.sample_rel_mse
        all_results.append(r)

        print(f"  std={r['std']:.4f} → {'spiky/VRE' if r['is_spiky'] else 'smooth/scalar'}")
        print(f"  relMSE={r['rel_mse']:.6f}  bps={r['bps']:.2f}")
        if r["method"] == "VRE":
            print(f"  unique_programs={r['unique_programs']}/{r['total_blocks']} ({100*r['unique_programs']/r['total_blocks']:.1f}%)")
            print(f"  avg_depth={r['avg_depth']:.2f}")
        print(f"  Block-RVQ baseline: {r['br_rmse']:.6f}")

    with open("results/m39_hybrid_encoder.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/m39_hybrid_encoder.json")


if __name__ == "__main__":
    main()
