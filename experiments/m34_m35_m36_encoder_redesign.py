#!/usr/bin/env python3
"""M34-M36: Encoder redesign prototypes on real reconstructed Llama 70B weights.

M34 Block-wise encoding: 4x4 / 8x8 / 16x16 blocks share a single scalar route value.
M35 Entropy-budget encoding: K-means VQ on greedy route values, hard K limit.
M36 Non-greedy encoder: Lloyd-Max scalar quantizer constrained to DRL route values.

GPU-accelerated where possible.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from block_vq import BlockRVQEncoding
from codebook import build_codebook
from route_encoder import decode_routes, encode_routes, rel_mse

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


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


def load_weight(enc_path: str, key: str) -> tuple[torch.Tensor, float]:
    enc = torch.load(enc_path, map_location="cpu")
    g = enc["encodings"][key]["groups"][0]
    bre = dict_to_bre(g)
    return bre.reconstruct(out_dtype=torch.float32), bre.sample_rel_mse


def m34_blockwise(w: torch.Tensor, ladder: torch.Tensor, block_size: int) -> dict:
    l_max = len(ladder)
    enc = encode_routes(w, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder.to(torch.float32)

    rows, cols = w.shape
    num_br = rows // block_size
    num_bc = cols // block_size
    w_blocks = (
        w.view(num_br, block_size, num_bc, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_br, num_bc, block_size * block_size)
    )

    K = route_values.numel()
    best_route = torch.empty(num_br, num_bc, dtype=torch.int64, device=w.device)
    batch = 128
    for br_start in range(0, num_br, batch):
        br_end = min(br_start + batch, num_br)
        wb = w_blocks[br_start:br_end]  # [B, num_bc, 16]
        diff = wb.unsqueeze(2) - route_values.view(1, 1, K, 1)
        sse = diff.square().sum(dim=-1)
        best_route[br_start:br_end] = sse.argmin(dim=-1)

    w_hat = (
        route_values[best_route]
        .unsqueeze(-1)
        .expand(num_br, num_bc, block_size * block_size)
        .reshape(num_br, num_bc, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )
    rmse = rel_mse(w, w_hat).item()
    unique_routes = torch.unique(best_route).numel()
    return {
        "block_size": block_size,
        "rel_mse": rmse,
        "unique_routes": int(unique_routes),
        "bps": math.log2(max(unique_routes, 1)),
    }


def m35_entropy_budget(w: torch.Tensor, ladder: torch.Tensor, K_target: int) -> dict:
    l_max = len(ladder)
    enc = encode_routes(w, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder.to(torch.float32)
    K = cb.keys.numel()
    freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()

    if K_target >= K:
        w_hat = decode_routes(enc, ladder)
        return {
            "K_target": K_target,
            "rel_mse": rel_mse(w, w_hat).item(),
            "K_actual": K,
            "bps": math.log2(K),
        }

    _, top_idx = freq.topk(K_target)
    centers = route_values[top_idx].clone()
    prev = None
    for _ in range(30):
        dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
        assignments = dist.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for c in range(K_target):
            mask = assignments == c
            if mask.any():
                wg = freq[mask]
                new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
        centers = new_centers
        if prev is not None and (assignments == prev).all():
            break
        prev = assignments

    w_flat = w.reshape(-1)
    w_assignments = torch.empty_like(w_flat, dtype=torch.int64)
    batch = 512 * 1024
    for start in range(0, w_flat.numel(), batch):
        end = min(start + batch, w_flat.numel())
        w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    w_hat = centers[w_assignments].reshape(w.shape)
    rmse = rel_mse(w, w_hat).item()
    return {
        "K_target": K_target,
        "rel_mse": rmse,
        "K_actual": K_target,
        "bps": math.log2(K_target),
    }


def m36_non_greedy(w: torch.Tensor, K_target: int, l_max: int = 5) -> dict:
    """Fast non-greedy: grid search on GPU, Lloyd-Max on sample."""
    w_flat = w.reshape(-1)
    perm = torch.randperm(w_flat.numel(), device=w.device)[:100_000]
    sample = w_flat[perm]

    best = None
    import itertools

    for start in [0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.0]:
        for decay in [0.5, 0.7, 0.8]:
            ladder = [start * (decay ** i) for i in range(l_max)]
            vals = set()
            for stop in range(1, l_max + 1):
                for digits in itertools.product([-1, 0, 1], repeat=stop):
                    vals.add(round(sum(d * ladder[i] for i, d in enumerate(digits)), 8))
            values = torch.tensor(sorted(vals), device=w.device, dtype=torch.float32)
            if values.numel() < K_target:
                continue

            sorted_s, _ = torch.sort(sample)
            step = max(1, len(sorted_s) // (K_target - 1))
            quantiles = sorted_s[::step][:K_target]
            centers = torch.zeros(K_target, device=w.device, dtype=torch.float32)
            for i, q in enumerate(quantiles):
                centers[i] = values[(values - q).abs().argmin()]

            prev_assign = None
            for _ in range(20):
                dist = (sample.unsqueeze(1) - centers.unsqueeze(0)).abs()
                assign = dist.argmin(dim=1)
                new_centers = torch.zeros_like(centers)
                changed = False
                for c in range(K_target):
                    mask = assign == c
                    if mask.any():
                        mean_val = sample[mask].mean()
                        idx = (values - mean_val).abs().argmin()
                        new_centers[c] = values[idx]
                        if values[idx] != centers[c]:
                            changed = True
                    else:
                        new_centers[c] = centers[c]
                centers = new_centers
                if not changed:
                    break
                if prev_assign is not None and (assign == prev_assign).all():
                    break
                prev_assign = assign

            assign_full = (w_flat.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
            w_hat = centers[assign_full].reshape(w.shape)
            rmse = rel_mse(w, w_hat).item()
            if best is None or rmse < best["rmse"]:
                best = {
                    "rmse": rmse,
                    "start": start,
                    "decay": decay,
                    "V": values.numel(),
                    "centers": [float(c) for c in centers.cpu()],
                    "ladder": ladder,
                }

    return {
        "K_target": K_target,
        "rel_mse": best["rmse"],
        "K_actual": K_target,
        "bps": math.log2(K_target),
        "best_ladder": best["ladder"],
        "best_start": best["start"],
        "best_decay": best["decay"],
        "unique_route_values": best["V"],
        "centers": best["centers"],
    }


def run_one(
    enc_path: str, key: str, out_path: str, fine_ladder: list[float], coarse_ladder: list[float]
) -> dict:
    w, br_rmse = load_weight(enc_path, key)
    w = w.to(DEVICE)
    row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w / row_scale

    results = {
        "layer": key,
        "shape": list(w.shape),
        "block_rvq_rel_mse": br_rmse,
        "experiments": {},
    }

    # Baseline greedy DRL v2 (fine ladder, no row norm)
    t0 = time.time()
    l_max = len(fine_ladder)
    ladder_t = torch.tensor(fine_ladder, device=DEVICE, dtype=torch.float32)
    enc = encode_routes(w, ladder_t, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    results["baseline_fine"] = {
        "rel_mse": rel_mse(w, decode_routes(enc, ladder_t)).item(),
        "K": cb.keys.numel(),
        "bps": math.log2(cb.keys.numel()),
        "time_s": round(time.time() - t0, 2),
    }

    # Baseline greedy DRL v2 (coarse ladder, row norm)
    t0 = time.time()
    l_max = len(coarse_ladder)
    ladder_t = torch.tensor(coarse_ladder, device=DEVICE, dtype=torch.float32)
    enc = encode_routes(w_norm, ladder_t, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    w_hat = decode_routes(enc, ladder_t) * row_scale
    results["baseline_coarse_rownorm"] = {
        "rel_mse": rel_mse(w, w_hat).item(),
        "K": cb.keys.numel(),
        "bps": math.log2(cb.keys.numel()),
        "time_s": round(time.time() - t0, 2),
    }

    # M34 block-wise
    t0 = time.time()
    results["m34_blockwise"] = {}
    for bs in [4, 8, 16]:
        r = m34_blockwise(w_norm, ladder_t, bs)
        results["m34_blockwise"][f"{bs}x{bs}"] = r
    results["m34_blockwise"]["time_s"] = round(time.time() - t0, 2)

    # M35 entropy budget
    t0 = time.time()
    results["m35_entropy_budget"] = {}
    for K in [8, 16, 32, 64, 128, 256]:
        r = m35_entropy_budget(w_norm, ladder_t, K)
        results["m35_entropy_budget"][f"K{K}"] = r
    results["m35_entropy_budget"]["time_s"] = round(time.time() - t0, 2)

    # M36 non-greedy
    t0 = time.time()
    results["m36_non_greedy"] = {}
    for K in [8, 16, 32]:
        r = m36_non_greedy(w_norm, K, l_max=5)
        results["m36_non_greedy"][f"K{K}"] = r
    results["m36_non_greedy"]["time_s"] = round(time.time() - t0, 2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    return results


def main():
    fine_ladder = [0.1 * (0.5 ** i) for i in range(8)]
    coarse_ladder = [1.0 * (0.5 ** i) for i in range(8)]

    configs = [
        (
            "results/m25_l54_q_gu_encodings.pt",
            "model.layers.54.self_attn.q_proj",
            "results/m34_m35_m36_l54_q.json",
        ),
        (
            "results/m25_l0_qkv_gu_encodings.pt",
            "model.layers.0.self_attn.q_proj",
            "results/m34_m35_m36_l0_q.json",
        ),
    ]

    for enc_path, key, out_path in configs:
        print(f"\n=== {key} ===")
        r = run_one(enc_path, key, out_path, fine_ladder, coarse_ladder)
        # Print concise summary
        print(f"Baseline fine: relMSE={r['baseline_fine']['rel_mse']:.6f} K={r['baseline_fine']['K']}")
        print(f"Baseline coarse+rn: relMSE={r['baseline_coarse_rownorm']['rel_mse']:.6f} K={r['baseline_coarse_rownorm']['K']}")
        for name in ["m34_blockwise", "m35_entropy_budget", "m36_non_greedy"]:
            print(f"{name}: { {k: f'{v['rel_mse']:.4f}/{v['bps']:.1f}b' if isinstance(v, dict) and 'rel_mse' in v else v for k, v in r[name].items()} }")


if __name__ == "__main__":
    main()
