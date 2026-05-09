"""M31 — Sparse Probe: block-sparsity analysis in Block-RVQ encodings.

Tests whether DRL v2 / Block-RVQ stage_id streams contain natural sparsity
that could be exploited by sparse matrix formats (block-CSR, BSR, etc.).

Metrics per layer:
- Zero-stage fraction: blocks where all stage_ids are zero (no contribution).
- Dominant-ID share: fraction of blocks covered by top-K IDs per stage.
- Tile occupancy: unique IDs per 64x64 / 128x128 tile.
- Row-wise entropy: H(stage_id distribution) per output row.
- ID persistence: correlation of IDs across adjacent positions.
"""
from __future__ import annotations

import json
import math
import sys
import time

import torch

sys.path.insert(0, ".")
from src import encoding_io


def entropy(counts: torch.Tensor, eps: float = 1e-12) -> float:
    p = counts.float() / counts.sum().clamp_min(eps)
    return -(p * torch.log2(p + eps)).sum().item()


def analyze_layer_fast(name: str, enc, tile_sizes: list[int] = [128, 256]) -> dict:
    g = enc.groups[0]
    num_stages = len(g.stage_ids)
    rows, blocks = g.stage_ids[0].shape
    block_size = g.block_size
    cols = blocks * block_size

    results = {
        "name": name,
        "shape": [rows, cols],
        "num_stages": num_stages,
        "block_size": block_size,
        "stages": [],
        "tiles": {},
    }

    for stage_idx in range(num_stages):
        ids = g.stage_ids[stage_idx]
        unique, counts = torch.unique(ids, return_counts=True)
        total = ids.numel()
        sorted_counts = counts.sort(descending=True).values
        top1 = sorted_counts[0].item() / total
        top8 = sorted_counts[:8].sum().item() / total if len(sorted_counts) >= 8 else 1.0
        top32 = sorted_counts[:32].sum().item() / total if len(sorted_counts) >= 32 else 1.0
        stage_res = {
            "unique_ids": int(unique.numel()),
            "top1_share": round(top1, 4),
            "top8_share": round(top8, 4),
            "top32_share": round(top32, 4),
            "entropy_bits": round(entropy(counts), 3),
        }
        results["stages"].append(stage_res)

    # Fast tile analysis: sample tiles, don't iterate all
    for tile_m in tile_sizes:
        for tile_n in tile_sizes:
            if tile_m > rows or tile_n > cols:
                continue
            tile_blocks_n = tile_n // block_size
            if tile_blocks_n == 0:
                continue

            num_tiles_m = rows // tile_m
            num_tiles_n = cols // tile_n
            if num_tiles_m == 0 or num_tiles_n == 0:
                continue

            # Sample up to 256 tiles deterministically
            max_sample = 256
            step_m = max(1, num_tiles_m // int(math.sqrt(max_sample)))
            step_n = max(1, num_tiles_n // int(math.sqrt(max_sample)))
            tile_uniques = []
            tile_top1_shares = []

            for tm in range(0, num_tiles_m, step_m):
                for tn in range(0, num_tiles_n, step_n):
                    if len(tile_uniques) >= max_sample:
                        break
                    m0 = tm * tile_m
                    n0 = tn * tile_blocks_n
                    m1 = m0 + tile_m
                    n1 = n0 + tile_blocks_n
                    # Collect IDs across stages for this tile
                    slices = [g.stage_ids[s][m0:m1, n0:n1].reshape(-1) for s in range(num_stages)]
                    tile_ids = torch.cat(slices)
                    unique_tile = torch.unique(tile_ids)
                    tile_uniques.append(int(unique_tile.numel()))
                    _, counts = torch.unique(tile_ids, return_counts=True)
                    tile_top1_shares.append(counts.max().item() / counts.sum().item())

            results["tiles"][f"{tile_m}x{tile_n}"] = {
                "avg_unique": round(sum(tile_uniques) / len(tile_uniques), 1),
                "max_unique": max(tile_uniques),
                "min_unique": min(tile_uniques),
                "avg_top1_share": round(sum(tile_top1_shares) / len(tile_top1_shares), 4),
                "sampled_tiles": len(tile_uniques),
            }

    ids0 = g.stage_ids[0]
    same_as_next = (ids0[:-1] == ids0[1:]).float().mean().item()
    results["row_persistence_stage0"] = round(same_as_next, 4)

    return results


def main():
    device = "cpu"
    print("=" * 80)
    print("M31 — Sparse Probe: Block-Sparsity Analysis")
    print("=" * 80)

    for path_name, pt_path in [
        ("layer54_q_gu", "results/m25_l54_q_gu_encodings.pt"),
        ("layer0_qkv_gu", "results/m25_l0_qkv_gu_encodings.pt"),
    ]:
        print(f"\n--- Loading {path_name} ---")
        enc_map = encoding_io.load_grouped_encoding_map(pt_path, device=device)
        all_results = {}
        for name, enc in enc_map.items():
            short_name = name.split(".")[-2] + "." + name.split(".")[-1]
            res = analyze_layer_fast(short_name, enc)
            all_results[short_name] = res

            print(f"\n{short_name} {res['shape']} {res['num_stages']} stages")
            print(f"  Row persistence (stage0): {res['row_persistence_stage0']}")
            for s in res["stages"]:
                print(f"  Stage: uniq={s['unique_ids']:4d} top1={s['top1_share']:.3f} top8={s['top8_share']:.3f} top32={s['top32_share']:.3f} H={s['entropy_bits']:.2f}b")
            for tile_key, tile_res in res["tiles"].items():
                print(f"  Tile {tile_key}: avg_unique={tile_res['avg_unique']:.1f} max={tile_res['max_unique']} top1={tile_res['avg_top1_share']:.3f}")

        out_json = f"results/m31_sparse_probe_{path_name}.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
