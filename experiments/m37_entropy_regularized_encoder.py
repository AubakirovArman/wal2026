"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M37: Entropy-regularized greedy encoder (vectorized 2-pass).

Pass 1: Standard greedy encode all weights → build codebook.
Pass 2: For each weight, re-assign to existing route minimizing MSE + regularization.

Regularization:
    cost = MSE(w, route) + lambda_new * is_new + lambda_rare / count[route]

Since Pass 2 only considers EXISTING routes (no new routes), lambda_new acts as
a hard constraint: if set to inf, no new routes are created beyond Pass 1 codebook.
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


def entropy_regularized_reassign(
    w: torch.Tensor,
    ladder: torch.Tensor,
    l_max: int,
    lambda_rare: float = 0.0,
    max_K: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """2-pass entropy-regularized encoder.

    Pass 1: Greedy encode.
    Pass 2: Reassign each weight to best existing route with regularization.
    If max_K is set, keep only top-K most frequent routes from Pass 1 and
    reassign all weights to these K routes.
    """
    # Pass 1: greedy
    enc = encode_routes(w, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)

    K = cb.keys.numel()
    route_values = (cb.digits.to(torch.float32) @ ladder).cpu()  # [K]

    # If max_K specified, reduce codebook to top-K frequent routes
    if max_K is not None and max_K < K:
        freq = torch.bincount(ids.reshape(-1).cpu().long(), minlength=K).float()
        _, keep_idx = freq.topk(max_K)
        keep_idx = keep_idx.sort().values  # sorted for determinism
        
        # Map old IDs to new IDs (or -1 for dropped)
        old_to_new = torch.full((K,), -1, dtype=torch.int64)
        old_to_new[keep_idx] = torch.arange(max_K)
        
        # Reassign all weights
        ids_flat = ids.reshape(-1).cpu().long()
        new_ids = old_to_new[ids_flat]
        
        # For weights assigned to dropped routes, reassign to nearest kept route
        dropped_mask = new_ids < 0
        if dropped_mask.any():
            w_flat = w.reshape(-1).cpu()
            kept_values = route_values[keep_idx]
            # Find nearest kept route for dropped weights
            dist = (w_flat[dropped_mask].unsqueeze(1) - kept_values.unsqueeze(0)).abs()
            nearest = dist.argmin(dim=1)
            new_ids[dropped_mask] = nearest
        
        ids = new_ids.reshape(ids.shape).to(w.device)
        route_values = kept_values.to(w.device)
        K = max_K
    else:
        route_values = route_values.to(w.device)
        # Pass 2: reassign with rare-route penalty
        if lambda_rare > 0:
            freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
            w_flat = w.reshape(-1)
            ids_flat = ids.reshape(-1)
            
            # Compute MSE to all routes batched
            batch = 256 * 1024
            new_ids = torch.empty_like(ids_flat)
            for start in range(0, w_flat.numel(), batch):
                end = min(start + batch, w_flat.numel())
                wb = w_flat[start:end].unsqueeze(1)  # [B, 1]
                diff = wb - route_values.unsqueeze(0)  # [B, K]
                mse = diff.square()
                reg = lambda_rare / freq.clamp_min(1.0)
                cost = mse + reg.unsqueeze(0)
                new_ids[start:end] = cost.argmin(dim=1)
            ids = new_ids.reshape(ids.shape)

    # Build digits and stop_depth from IDs
    w_hat = route_values[ids]
    rmse = rel_mse(w, w_hat).item()
    
    freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
    entropy = -(freq / freq.sum() * torch.log2(freq / freq.sum()).clamp_min(1e-12)).sum().item()
    
    stats = {
        "unique_routes": K,
        "entropy_bits": entropy,
        "rel_mse": rmse,
        "bps": math.log2(K),
    }
    return ids, route_values, stats


def test_layer(enc_path, key, coarse_ladder, configs):
    enc = torch.load(enc_path, map_location="cpu")
    g = enc["encodings"][key]["groups"][0]
    bre = dict_to_bre(g)
    w = bre.reconstruct(out_dtype=torch.float32).to(DEVICE)
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale

    l_max = len(coarse_ladder)
    ladder = torch.tensor(coarse_ladder, device=DEVICE, dtype=torch.float32)

    results = {
        "layer": key,
        "shape": list(w.shape),
        "br_rmse": bre.sample_rel_mse,
        "configs": [],
    }

    for cfg in configs:
        max_K = cfg.get("max_K")
        lambda_rare = cfg.get("lambda_rare", 0.0)
        _, _, stats = entropy_regularized_reassign(w_norm, ladder, l_max, lambda_rare=lambda_rare, max_K=max_K)
        stats["config"] = cfg
        results["configs"].append(stats)
        print(
            f"  {cfg}: relMSE={stats['rel_mse']:.6f}, K={stats['unique_routes']}, "
            f"entropy={stats['entropy_bits']:.2f}b, bps={stats['bps']:.2f}"
        )

    return results


def main():
    coarse_ladder = [1.0 * (0.5 ** i) for i in range(8)]
    configs = [
        {"max_K": None, "lambda_rare": 0.0},  # Baseline
        {"max_K": 256, "lambda_rare": 0.0},
        {"max_K": 128, "lambda_rare": 0.0},
        {"max_K": 64, "lambda_rare": 0.0},
        {"max_K": 32, "lambda_rare": 0.0},
        {"max_K": 16, "lambda_rare": 0.0},
        {"max_K": 8, "lambda_rare": 0.0},
        {"max_K": None, "lambda_rare": 1e-4},
        {"max_K": None, "lambda_rare": 1e-3},
    ]

    test_configs = [
        ("results/m25_l54_q_gu_encodings.pt", "model.layers.54.self_attn.q_proj"),
        ("results/m25_l0_qkv_gu_encodings.pt", "model.layers.0.self_attn.q_proj"),
    ]

    all_results = []
    for enc_path, key in test_configs:
        print(f"\n=== {key} ===")
        r = test_layer(enc_path, key, coarse_ladder, configs)
        all_results.append(r)

    with open("results/m37_entropy_regularized.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/m37_entropy_regularized.json")


if __name__ == "__main__":
    main()
