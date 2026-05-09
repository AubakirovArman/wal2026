"""M33 — Encoder Prototype: Program-Cost Regularized Route Encoding.

Demonstrates that post-hoc smoothing of the ID field can reduce program cost
(unique routes, entropy) at a modest reconstruction penalty.

Method:
1. Standard greedy encode (baseline)
2. Greedy horizontal smoothing: scan each row left-to-right.
   If using the previous element's route doesn't increase error by > threshold,
   replace current element's route with previous element's route.

Metrics: relMSE, unique IDs, entropy, smoothness, tile occupancy.
"""
from __future__ import annotations

import json
import time

import torch
from torch import Tensor

import sys
sys.path.insert(0, ".")
from src import route_encoder, codebook


def program_cost(ids: Tensor, w: Tensor, w_hat: Tensor, tiles_m: int = 64, tiles_n: int = 64) -> dict:
    n, k = ids.shape
    uniq, counts = torch.unique(ids, return_counts=True)
    total = ids.numel()
    p = counts.float() / total
    entropy = -(p * torch.log2(p + 1e-12)).sum().item()

    h_same = (ids[:, :-1] == ids[:, 1:]).float().mean().item()
    v_same = (ids[:-1, :] == ids[1:, :]).float().mean().item()

    tile_uniques = []
    tile_top1 = []
    for tm in range(0, n, tiles_m):
        for tn in range(0, k, tiles_n):
            tile = ids[tm:tm+tiles_m, tn:tn+tiles_n]
            u, c = torch.unique(tile, return_counts=True)
            tile_uniques.append(int(u.numel()))
            tile_top1.append((c.max().item() / c.sum().item()) if c.numel() > 0 else 1.0)

    rel_mse = ((w - w_hat) ** 2).mean() / (w ** 2).mean()

    return {
        "unique_ids": int(uniq.numel()),
        "entropy_bits": round(entropy, 3),
        "h_smoothness": round(h_same, 4),
        "v_smoothness": round(v_same, 4),
        "avg_tile_unique": round(sum(tile_uniques) / len(tile_uniques), 1),
        "avg_tile_top1": round(sum(tile_top1) / len(tile_top1), 4),
        "rel_mse": round(rel_mse.item(), 6),
    }


def horizontal_smooth(
    w: Tensor,
    ids: Tensor,
    codebook_sum: Tensor,
    row_scale: Tensor,
    rel_threshold: float = 1.01,  # allow 1% increase in single-weight squared error
) -> Tensor:
    """Greedy horizontal smoothing: if neighbor's route is almost as good, use it."""
    n, k = ids.shape
    ids_new = ids.clone()
    w_hat = codebook_sum[ids.long()] * row_scale.unsqueeze(1)
    changed = 0

    for i in range(n):
        for j in range(1, k):
            orig_id = ids_new[i, j].item()
            prev_id = ids_new[i, j - 1].item()
            if orig_id == prev_id:
                continue
            orig_val = codebook_sum[orig_id].item()
            prev_val = codebook_sum[prev_id].item()
            w_ij = w[i, j].item()
            err_orig = (orig_val - w_ij) ** 2
            err_prev = (prev_val - w_ij) ** 2
            if err_prev <= err_orig * rel_threshold:
                ids_new[i, j] = prev_id
                changed += 1

    return ids_new, changed


def vertical_smooth(
    w: Tensor,
    ids: Tensor,
    codebook_sum: Tensor,
    row_scale: Tensor,
    rel_threshold: float = 1.01,
) -> Tensor:
    """Greedy vertical smoothing."""
    n, k = ids.shape
    ids_new = ids.clone()
    changed = 0

    for i in range(1, n):
        for j in range(k):
            orig_id = ids_new[i, j].item()
            prev_id = ids_new[i - 1, j].item()
            if orig_id == prev_id:
                continue
            orig_val = codebook_sum[orig_id].item()
            prev_val = codebook_sum[prev_id].item()
            w_ij = w[i, j].item()
            err_orig = (orig_val - w_ij) ** 2
            err_prev = (prev_val - w_ij) ** 2
            if err_prev <= err_orig * rel_threshold:
                ids_new[i, j] = prev_id
                changed += 1

    return ids_new, changed


def quantize_to_topk(
    w: Tensor,
    ids: Tensor,
    codebook_sum: Tensor,
    row_scale: Tensor,
    k: int = 256,
    rel_threshold: float = 1.5,
) -> Tensor:
    """Replace each weight's route with top-K most frequent routes if error is acceptable."""
    uniq, counts = torch.unique(ids, return_counts=True)
    topk_ids = uniq[counts.argsort(descending=True)[:k]]
    topk_set = set(topk_ids.tolist())

    ids_new = ids.clone()
    changed = 0
    n, k_dim = ids.shape

    for i in range(n):
        for j in range(k_dim):
            orig_id = ids_new[i, j].item()
            if orig_id in topk_set:
                continue
            orig_val = codebook_sum[orig_id].item()
            w_ij = w[i, j].item()
            err_orig = (orig_val - w_ij) ** 2
            best_id = orig_id
            best_err = err_orig
            for cand in topk_ids.tolist():
                cand_val = codebook_sum[cand].item()
                err_cand = (cand_val - w_ij) ** 2
                if err_cand < best_err * rel_threshold and err_cand < best_err:
                    best_id = cand
                    best_err = err_cand
            if best_id != orig_id:
                ids_new[i, j] = best_id
                changed += 1

    return ids_new, changed


def main():
    print("=" * 80)
    print("M33 — Encoder Prototype: Program-Cost Regularizer")
    print("=" * 80)

    device = "cpu"
    dtype = torch.float32
    n, k = 256, 256
    l_max = 12

    torch.manual_seed(42)
    w = torch.randn(n, k, dtype=dtype, device=device)
    ladder = torch.tensor([1.0 / (2 ** i) for i in range(l_max)], device=device, dtype=dtype)
    row_scale = torch.ones(n, dtype=dtype, device=device)

    # Baseline
    t0 = time.time()
    enc = route_encoder.encode_routes(w, ladder, l_max=l_max)
    cb, ids = codebook.build_codebook(enc.digits, enc.stop_depth, l_max)
    codebook_sum = (cb.digits.to(dtype) * ladder[:l_max].unsqueeze(0)).sum(dim=1)
    w_hat = codebook_sum[ids.long()] * row_scale.unsqueeze(1)
    t_enc = time.time() - t0

    print(f"\nStandard encode: {t_enc:.3f}s")
    baseline = program_cost(ids, w, w_hat)
    print("Baseline:", json.dumps(baseline))

    results = {"baseline": baseline, "variants": []}

    # Variant 1: Horizontal smooth (1% threshold)
    ids_h, changed_h = horizontal_smooth(w, ids, codebook_sum, row_scale, rel_threshold=1.01)
    w_hat_h = codebook_sum[ids_h.long()] * row_scale.unsqueeze(1)
    pc_h = program_cost(ids_h, w, w_hat_h)
    pc_h["name"] = "h_smooth_1pct"
    pc_h["changed"] = changed_h
    results["variants"].append(pc_h)
    print(f"\nH-smooth 1%:  changed={changed_h}  {json.dumps(pc_h)}")

    # Variant 2: Horizontal smooth (5% threshold)
    ids_h5, changed_h5 = horizontal_smooth(w, ids, codebook_sum, row_scale, rel_threshold=1.05)
    w_hat_h5 = codebook_sum[ids_h5.long()] * row_scale.unsqueeze(1)
    pc_h5 = program_cost(ids_h5, w, w_hat_h5)
    pc_h5["name"] = "h_smooth_5pct"
    pc_h5["changed"] = changed_h5
    results["variants"].append(pc_h5)
    print(f"H-smooth 5%:  changed={changed_h5}  {json.dumps(pc_h5)}")

    # Variant 3: Vertical + Horizontal (5% each)
    ids_v, changed_v = vertical_smooth(w, ids_h5, codebook_sum, row_scale, rel_threshold=1.05)
    w_hat_v = codebook_sum[ids_v.long()] * row_scale.unsqueeze(1)
    pc_v = program_cost(ids_v, w, w_hat_v)
    pc_v["name"] = "hv_smooth_5pct"
    pc_v["changed"] = changed_v
    results["variants"].append(pc_v)
    print(f"HV-smooth 5%: changed={changed_v}  {json.dumps(pc_v)}")

    # Variant 4: Aggressive horizontal smooth (2x threshold)
    ids_h2, changed_h2 = horizontal_smooth(w, ids, codebook_sum, row_scale, rel_threshold=2.0)
    w_hat_h2 = codebook_sum[ids_h2.long()] * row_scale.unsqueeze(1)
    pc_h2 = program_cost(ids_h2, w, w_hat_h2)
    pc_h2["name"] = "h_smooth_2x"
    pc_h2["changed"] = changed_h2
    results["variants"].append(pc_h2)
    print(f"H-smooth 2x:   changed={changed_h2}  {json.dumps(pc_h2)}")

    # Variant 5: Tile-wise majority vote (16x16 tiles)
    ids_tile = ids.clone()
    tile_m = tile_n = 16
    num_tm = n // tile_m
    num_tn = k // tile_n
    changed_tile = 0
    for tm in range(num_tm):
        for tn in range(num_tn):
            tile = ids[tm*tile_m:(tm+1)*tile_m, tn*tile_n:(tn+1)*tile_n]
            uniq, counts = torch.unique(tile, return_counts=True)
            mode_id = uniq[counts.argmax()].item()
            tile_orig = ids_tile[tm*tile_m:(tm+1)*tile_m, tn*tile_n:(tn+1)*tile_n].clone()
            ids_tile[tm*tile_m:(tm+1)*tile_m, tn*tile_n:(tn+1)*tile_n] = mode_id
            changed_tile += (tile_orig != mode_id).sum().item()
    w_hat_tile = codebook_sum[ids_tile.long()] * row_scale.unsqueeze(1)
    pc_tile = program_cost(ids_tile, w, w_hat_tile)
    pc_tile["name"] = "tile_majority_16"
    pc_tile["changed"] = changed_tile
    results["variants"].append(pc_tile)
    print(f"Tile majority: changed={changed_tile}  {json.dumps(pc_tile)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<20} {'relMSE':>10} {'unique':>8} {'entropy':>8} {'h_smooth':>10} {'tile_uniq':>10}")
    print(f"{'baseline':<20} {baseline['rel_mse']:>10.6f} {baseline['unique_ids']:>8} {baseline['entropy_bits']:>8.2f} {baseline['h_smoothness']:>10.4f} {baseline['avg_tile_unique']:>10.1f}")
    for v in results["variants"]:
        print(f"{v['name']:<20} {v['rel_mse']:>10.6f} {v['unique_ids']:>8} {v['entropy_bits']:>8.2f} {v['h_smoothness']:>10.4f} {v['avg_tile_unique']:>10.1f}")

    with open("results/m33_encoder_program_cost.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/m33_encoder_program_cost.json")


if __name__ == "__main__":
    main()
