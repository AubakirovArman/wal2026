"""Probe a graph-style shared PQ codebook across multiple weight tensors.

Hypothesis (route-graph idea):
- Currently each tensor has its own per-stage PQ codebooks.
- Many block "motifs" recur across tensors (q vs k vs v of the same layer,
  same projection in adjacent layers, etc.).
- A graph-DB style representation would store one shared dictionary of motifs
  and let every tensor just point into it (= edges into shared nodes).

This probe measures, per product split and stage, how much storage and
quality changes when the codebook is shared across a chosen group of
tensors versus encoded per-tensor.

Run:
  python dwl2_dynamic_route/experiments/m13a_shared_codebook_graph_probe.py \
      --tensor-names model.language_model.layers.54.self_attn.q_proj.weight,model.language_model.layers.54.self_attn.k_proj.weight \
      --block-size 32 --codebook-size 256 --num-stages 3 --product-splits 4 \
      --out dwl2_dynamic_route/results/m13a_l54_qk_shared.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import (  # noqa: E402
    _assign_to_codebook,
    _fit_kmeans,
    _id_storage_bytes,
    _reshape_blocks,
)


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]


def load_weight(name: str, device: str) -> torch.Tensor:
    shard = MODEL_DIR / INDEX[name]
    with safe_open(str(shard), framework="pt", device=device) as handle:
        return handle.get_tensor(name).to(torch.bfloat16)


def normalize_and_block(weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (blocks[N_blocks, block_size], row_scale[rows,1], rows, padded_cols)."""
    row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(torch.float16)
    w_norm = weight.to(torch.float32) / row_scale.to(torch.float32)
    blocks, rows, padded_cols = _reshape_blocks(w_norm, block_size)
    return blocks, row_scale, rows, padded_cols


def encode_per_tensor_pq(
    blocks: torch.Tensor,
    *,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    sample_limit: int,
    kmeans_iters: int,
    batch_size: int,
) -> tuple[torch.Tensor, list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Independent per-tensor PQ + RVQ. Returns (recon, codebooks_per_split, ids_per_split).

    codebooks_per_split[s][stage] : Tensor[codebook_size, sub_dim]
    ids_per_split[s][stage]       : Tensor[N_blocks] long
    """
    sub_dim = block_size // product_splits
    n_blocks = int(blocks.shape[0])
    recon = torch.zeros_like(blocks, dtype=torch.float32)

    codebooks_per_split: list[list[torch.Tensor]] = []
    ids_per_split: list[list[torch.Tensor]] = []

    generator = torch.Generator(device=blocks.device)
    generator.manual_seed(0)

    for split in range(product_splits):
        lo = split * sub_dim
        hi = lo + sub_dim
        residual_slice = blocks[:, lo:hi].clone().to(torch.float32)
        sample_count = min(int(sample_limit), n_blocks)
        sample_count = max(sample_count, codebook_size)
        sample_count = min(sample_count, n_blocks)
        pick = torch.randperm(n_blocks, generator=generator, device=blocks.device)[:sample_count]
        sample_slice = residual_slice[pick].clone()

        stage_codebooks: list[torch.Tensor] = []
        stage_ids: list[torch.Tensor] = []
        for _ in range(num_stages):
            codebook = _fit_kmeans(sample_slice, codebook_size=codebook_size, iters=kmeans_iters, batch_size=batch_size)
            full_ids, _ = _assign_to_codebook(residual_slice, codebook, batch_size=batch_size)
            residual_slice -= codebook[full_ids]
            sample_ids, _ = _assign_to_codebook(sample_slice, codebook, batch_size=batch_size)
            sample_slice -= codebook[sample_ids]
            stage_codebooks.append(codebook)
            stage_ids.append(full_ids)
        # accumulated approximation for this split
        approx_slice = torch.zeros_like(residual_slice)
        for cb, ids in zip(stage_codebooks, stage_ids):
            approx_slice += cb[ids]
        recon[:, lo:hi] = approx_slice

        codebooks_per_split.append(stage_codebooks)
        ids_per_split.append(stage_ids)

    return recon, codebooks_per_split, ids_per_split


def encode_shared_pq(
    blocks_per_tensor: list[torch.Tensor],
    *,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    sample_per_tensor: int,
    kmeans_iters: int,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], list[list[list[torch.Tensor]]]]:
    """Joint PQ where each (split, stage) codebook is fit on pooled samples
    from all tensors and then used to assign ids per tensor.

    Returns (recon_per_tensor, shared_codebooks_per_split, ids_per_tensor_per_split)
      shared_codebooks_per_split[s][stage] : Tensor[codebook_size, sub_dim]
      ids_per_tensor_per_split[t][s][stage] : Tensor[N_blocks_t]
    """
    sub_dim = block_size // product_splits

    # Per-tensor running residual slices (one per tensor, will mutate stage by stage).
    residual_per_tensor_per_split: list[list[torch.Tensor]] = []
    for blocks in blocks_per_tensor:
        per_split = []
        for split in range(product_splits):
            lo = split * sub_dim
            hi = lo + sub_dim
            per_split.append(blocks[:, lo:hi].clone().to(torch.float32))
        residual_per_tensor_per_split.append(per_split)

    shared_codebooks_per_split: list[list[torch.Tensor]] = []
    ids_per_tensor_per_split: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(product_splits)] for _ in blocks_per_tensor
    ]

    generator = torch.Generator(device=blocks_per_tensor[0].device)
    generator.manual_seed(0)

    for split in range(product_splits):
        stage_codebooks: list[torch.Tensor] = []
        for _ in range(num_stages):
            # Pool a sample of residual blocks across all tensors for this split.
            pooled_samples = []
            for t, per_split in enumerate(residual_per_tensor_per_split):
                blocks_slice = per_split[split]
                n = int(blocks_slice.shape[0])
                k = min(int(sample_per_tensor), n)
                pick = torch.randperm(n, generator=generator, device=blocks_slice.device)[:k]
                pooled_samples.append(blocks_slice[pick])
            pooled = torch.cat(pooled_samples, dim=0)
            if pooled.shape[0] < codebook_size:
                # Make sure k-means has enough points by replicating if needed.
                reps = math.ceil(codebook_size / pooled.shape[0])
                pooled = pooled.repeat(reps, 1)[: max(codebook_size, pooled.shape[0])]
            codebook = _fit_kmeans(
                pooled,
                codebook_size=codebook_size,
                iters=kmeans_iters,
                batch_size=batch_size,
            )
            stage_codebooks.append(codebook)
            # Assign each tensor's residual slice to the SHARED codebook.
            for t, per_split in enumerate(residual_per_tensor_per_split):
                blocks_slice = per_split[split]
                full_ids, _ = _assign_to_codebook(blocks_slice, codebook, batch_size=batch_size)
                ids_per_tensor_per_split[t][split].append(full_ids)
                per_split[split] = blocks_slice - codebook[full_ids]
        shared_codebooks_per_split.append(stage_codebooks)

    # Reconstruct each tensor.
    recon_per_tensor: list[torch.Tensor] = []
    for t, blocks in enumerate(blocks_per_tensor):
        recon = torch.zeros_like(blocks, dtype=torch.float32)
        for split in range(product_splits):
            lo = split * sub_dim
            hi = lo + sub_dim
            approx_slice = torch.zeros(blocks.shape[0], sub_dim, dtype=torch.float32, device=blocks.device)
            for stage_idx, codebook in enumerate(shared_codebooks_per_split[split]):
                ids = ids_per_tensor_per_split[t][split][stage_idx]
                approx_slice += codebook[ids]
            recon[:, lo:hi] = approx_slice
        recon_per_tensor.append(recon)

    return recon_per_tensor, shared_codebooks_per_split, ids_per_tensor_per_split


def per_tensor_storage_bytes(
    ids_per_split: list[list[torch.Tensor]],
    codebooks_per_split: list[list[torch.Tensor]],
    row_scale_numel: int,
    sub_dim: int,
    codebook_size: int,
) -> dict[str, int]:
    id_bytes_each = _id_storage_bytes(codebook_size)
    ids_total = 0
    for stages_ids in ids_per_split:
        for ids in stages_ids:
            ids_total += int(ids.numel()) * id_bytes_each
    cb_total = 0
    for stages_cb in codebooks_per_split:
        for cb in stages_cb:
            # stored as fp16 [codebook_size, sub_dim]
            cb_total += int(cb.shape[0]) * int(cb.shape[1]) * 2
    row_scale_bytes = row_scale_numel * 2  # fp16
    total = ids_total + cb_total + row_scale_bytes
    return {
        "ids_bytes": ids_total,
        "codebook_bytes": cb_total,
        "row_scale_bytes": row_scale_bytes,
        "total_bytes": total,
    }


def shared_storage_bytes(
    ids_per_tensor_per_split: list[list[list[torch.Tensor]]],
    shared_codebooks_per_split: list[list[torch.Tensor]],
    row_scale_numel_per_tensor: list[int],
    sub_dim: int,
    codebook_size: int,
) -> dict[str, int]:
    id_bytes_each = _id_storage_bytes(codebook_size)
    ids_total = 0
    for per_tensor in ids_per_tensor_per_split:
        for stages_ids in per_tensor:
            for ids in stages_ids:
                ids_total += int(ids.numel()) * id_bytes_each
    cb_total = 0
    for stages_cb in shared_codebooks_per_split:
        for cb in stages_cb:
            cb_total += int(cb.shape[0]) * int(cb.shape[1]) * 2
    row_scale_bytes = sum(row_scale_numel_per_tensor) * 2
    total = ids_total + cb_total + row_scale_bytes
    return {
        "ids_bytes": ids_total,
        "shared_codebook_bytes": cb_total,
        "row_scale_bytes": row_scale_bytes,
        "total_bytes": total,
    }


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).square().mean()
    den = b.float().square().mean().clamp_min(1e-12)
    return float((diff / den).item())


def storage_megabytes(num_bytes: int) -> float:
    return float(num_bytes) / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-names",
        default=",".join(
            [
                "model.language_model.layers.54.self_attn.q_proj.weight",
                "model.language_model.layers.54.self_attn.k_proj.weight",
            ]
        ),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--sample-per-tensor", type=int, default=32_768)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--out",
        default=str(ROOT / "dwl2_dynamic_route/results/m13a_shared_codebook_graph_probe.json"),
    )
    args = parser.parse_args()

    tensor_names = [p.strip() for p in args.tensor_names.split(",") if p.strip()]
    if args.block_size % args.product_splits != 0:
        raise ValueError("product_splits must divide block_size")
    sub_dim = args.block_size // args.product_splits

    blocks_per_tensor: list[torch.Tensor] = []
    row_scale_per_tensor: list[torch.Tensor] = []
    rows_per_tensor: list[int] = []
    padded_cols_per_tensor: list[int] = []
    weight_per_tensor: list[torch.Tensor] = []

    for name in tensor_names:
        weight = load_weight(name, args.device)
        weight_per_tensor.append(weight)
        blocks, row_scale, rows, padded_cols = normalize_and_block(weight, args.block_size)
        blocks_per_tensor.append(blocks)
        row_scale_per_tensor.append(row_scale)
        rows_per_tensor.append(rows)
        padded_cols_per_tensor.append(padded_cols)
        print(
            f"loaded {name}: shape={tuple(weight.shape)} blocks={tuple(blocks.shape)} rows={rows}",
            flush=True,
        )

    # ---- per-tensor independent encoding ----
    per_tensor_summaries = []
    indep_total_bytes = 0
    indep_total_w_bytes = 0
    indep_total_w_err = 0.0
    indep_total_w_den = 0.0
    print("\n=== INDEPENDENT PER-TENSOR PQ ===", flush=True)
    for i, name in enumerate(tensor_names):
        blocks = blocks_per_tensor[i]
        weight = weight_per_tensor[i]
        rows = rows_per_tensor[i]
        padded_cols = padded_cols_per_tensor[i]
        row_scale = row_scale_per_tensor[i]
        t0 = time.time()
        recon_blocks, codebooks_per_split, ids_per_split = encode_per_tensor_pq(
            blocks,
            block_size=args.block_size,
            codebook_size=args.codebook_size,
            num_stages=args.num_stages,
            product_splits=args.product_splits,
            sample_limit=args.sample_per_tensor,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
        recon_w_norm = recon_blocks.view(rows, padded_cols)[:, : weight.shape[1]]
        recon_w = recon_w_norm * row_scale.to(torch.float32)
        err = rel_mse(recon_w, weight)
        # Storage
        storage = per_tensor_storage_bytes(
            ids_per_split,
            codebooks_per_split,
            row_scale_numel=int(row_scale.numel()),
            sub_dim=sub_dim,
            codebook_size=args.codebook_size,
        )
        bits_per_w = 8.0 * storage["total_bytes"] / max(int(weight.numel()), 1)
        dt = time.time() - t0
        diff_sq = (recon_w.float() - weight.float()).square().sum().item()
        den_sq = weight.float().square().sum().item()
        indep_total_w_err += diff_sq
        indep_total_w_den += den_sq
        indep_total_bytes += storage["total_bytes"]
        indep_total_w_bytes += int(weight.numel()) * 2
        print(
            f"  {name}: rel_mse={err:.3e} storage_mb={storage_megabytes(storage['total_bytes']):.2f} bits/w={bits_per_w:.3f} ({dt:.1f}s)",
            flush=True,
        )
        # ----- id-tuple uniqueness analytics (graph-edge potential) -----
        # Stack ids as [N_blocks, num_splits * num_stages] then count unique rows.
        all_stage_ids = []
        for stages_ids in ids_per_split:
            for ids in stages_ids:
                all_stage_ids.append(ids.to(torch.int32))
        stacked_ids = torch.stack(all_stage_ids, dim=1)  # [N_blocks, S*K]
        n_blocks_total = int(stacked_ids.shape[0])
        unique_full, _ = torch.unique(stacked_ids, dim=0, return_inverse=True)
        n_unique_full = int(unique_full.shape[0])
        # Per-split tuple uniqueness (each split independently).
        per_split_unique_counts = []
        sub_count = args.num_stages
        for split in range(args.product_splits):
            split_block = stacked_ids[:, split * sub_count : (split + 1) * sub_count]
            unique_split, _ = torch.unique(split_block, dim=0, return_inverse=True)
            per_split_unique_counts.append(int(unique_split.shape[0]))
        full_uniqueness_ratio = n_unique_full / max(n_blocks_total, 1)
        # ----- per-stage id entropy (Shannon, in bits) -----
        per_stage_entropies = []
        for ids in all_stage_ids:
            counts = torch.bincount(ids.to(torch.int64), minlength=args.codebook_size).to(torch.float64)
            probs = counts / counts.sum().clamp_min(1.0)
            nonzero = probs > 0
            entropy = float(-(probs[nonzero] * probs[nonzero].log2()).sum().item())
            per_stage_entropies.append(entropy)
        avg_id_entropy_bits = sum(per_stage_entropies) / max(len(per_stage_entropies), 1)
        raw_id_bits = 8.0 * _id_storage_bytes(args.codebook_size)
        entropy_compression = avg_id_entropy_bits / max(raw_id_bits, 1e-9)
        print(
            f"    id-tuples: full unique {n_unique_full}/{n_blocks_total} ({full_uniqueness_ratio*100:.2f}%) "
            f"per-split unique {per_split_unique_counts}\n"
            f"    id-entropy: avg={avg_id_entropy_bits:.3f} bits (raw={raw_id_bits:.0f}) "
            f"compression_potential={entropy_compression*100:.1f}% of raw",
            flush=True,
        )

        per_tensor_summaries.append(
            {
                "name": name,
                "indep_rel_mse": err,
                "indep_storage_mb": storage_megabytes(storage["total_bytes"]),
                "indep_storage_bytes": storage["total_bytes"],
                "indep_codebook_bytes": storage["codebook_bytes"],
                "indep_ids_bytes": storage["ids_bytes"],
                "indep_bits_per_weight": bits_per_w,
                "n_blocks": n_blocks_total,
                "n_unique_full_id_tuple": n_unique_full,
                "full_id_tuple_uniqueness": full_uniqueness_ratio,
                "per_split_unique_counts": per_split_unique_counts,
                "per_stage_id_entropy_bits": per_stage_entropies,
                "avg_id_entropy_bits": avg_id_entropy_bits,
                "entropy_compression_fraction": entropy_compression,
            }
        )

    indep_pooled_rel_mse = indep_total_w_err / max(indep_total_w_den, 1e-12)
    print(
        f"\nindep total: storage_mb={storage_megabytes(indep_total_bytes):.2f} "
        f"pooled_rel_mse={indep_pooled_rel_mse:.3e}",
        flush=True,
    )

    # ---- shared codebook encoding ----
    print("\n=== SHARED-CODEBOOK PQ (graph-style) ===", flush=True)
    t1 = time.time()
    recon_blocks_per_tensor, shared_codebooks_per_split, ids_per_tensor_per_split = encode_shared_pq(
        blocks_per_tensor,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        num_stages=args.num_stages,
        product_splits=args.product_splits,
        sample_per_tensor=args.sample_per_tensor,
        kmeans_iters=args.kmeans_iters,
        batch_size=args.batch_size,
    )
    dt_shared = time.time() - t1

    shared_total_bytes = 0
    shared_total_w_err = 0.0
    shared_total_w_den = 0.0
    storage = shared_storage_bytes(
        ids_per_tensor_per_split,
        shared_codebooks_per_split,
        row_scale_numel_per_tensor=[int(rs.numel()) for rs in row_scale_per_tensor],
        sub_dim=sub_dim,
        codebook_size=args.codebook_size,
    )
    shared_total_bytes = storage["total_bytes"]
    for i, name in enumerate(tensor_names):
        weight = weight_per_tensor[i]
        rows = rows_per_tensor[i]
        padded_cols = padded_cols_per_tensor[i]
        row_scale = row_scale_per_tensor[i]
        recon_w_norm = recon_blocks_per_tensor[i].view(rows, padded_cols)[:, : weight.shape[1]]
        recon_w = recon_w_norm * row_scale.to(torch.float32)
        err = rel_mse(recon_w, weight)
        diff_sq = (recon_w.float() - weight.float()).square().sum().item()
        den_sq = weight.float().square().sum().item()
        shared_total_w_err += diff_sq
        shared_total_w_den += den_sq
        per_tensor_summaries[i]["shared_rel_mse"] = err
        print(
            f"  {name}: shared rel_mse={err:.3e} (vs indep {per_tensor_summaries[i]['indep_rel_mse']:.3e})",
            flush=True,
        )
    shared_pooled_rel_mse = shared_total_w_err / max(shared_total_w_den, 1e-12)

    bits_per_w_shared = 8.0 * shared_total_bytes / max(sum(int(w.numel()) for w in weight_per_tensor), 1)
    print(
        f"\nshared total: storage_mb={storage_megabytes(shared_total_bytes):.2f} "
        f"pooled_rel_mse={shared_pooled_rel_mse:.3e} bits/w={bits_per_w_shared:.3f} ({dt_shared:.1f}s)",
        flush=True,
    )

    storage_savings = 1.0 - shared_total_bytes / max(indep_total_bytes, 1)
    rel_mse_delta = shared_pooled_rel_mse - indep_pooled_rel_mse
    print(
        f"\n=== GRAPH SHARING DELTA ===\n"
        f"  storage indep:  {storage_megabytes(indep_total_bytes):.2f} MB  pooled rel_mse {indep_pooled_rel_mse:.3e}\n"
        f"  storage shared: {storage_megabytes(shared_total_bytes):.2f} MB  pooled rel_mse {shared_pooled_rel_mse:.3e}\n"
        f"  storage savings: {storage_savings*100:.2f}%   rel_mse delta: {rel_mse_delta:+.3e}",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "tensor_names": tensor_names,
                "cfg": {
                    "block_size": args.block_size,
                    "codebook_size": args.codebook_size,
                    "num_stages": args.num_stages,
                    "product_splits": args.product_splits,
                    "sample_per_tensor": args.sample_per_tensor,
                    "kmeans_iters": args.kmeans_iters,
                },
                "per_tensor": per_tensor_summaries,
                "indep_total_storage_mb": storage_megabytes(indep_total_bytes),
                "indep_pooled_rel_mse": indep_pooled_rel_mse,
                "shared_total_storage_mb": storage_megabytes(shared_total_bytes),
                "shared_pooled_rel_mse": shared_pooled_rel_mse,
                "shared_bits_per_weight": bits_per_w_shared,
                "storage_savings_fraction": storage_savings,
                "rel_mse_delta": rel_mse_delta,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
