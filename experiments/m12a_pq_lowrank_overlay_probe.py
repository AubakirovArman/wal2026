from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import encode_grouped_block_residual_vq


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
DEFAULT_TENSOR_NAMES = (
    "model.layers.54.self_attn.q_proj.weight",
    "model.layers.54.self_attn.k_proj.weight",
    "model.layers.54.self_attn.v_proj.weight",
    "model.layers.54.mlp.gate_proj.weight",
)


def load_weight(name: str, device: str) -> torch.Tensor:
    shard = MODEL_DIR / INDEX[name]
    with safe_open(str(shard), framework="pt", device=device) as handle:
        return handle.get_tensor(name).to(torch.bfloat16)


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).square().mean()
    den = b.float().square().mean().clamp_min(1e-12)
    return float((diff / den).item())


def storage_megabytes(num_bytes: int) -> float:
    return float(num_bytes) / 2**20


def dense_bf16_storage_bytes(weight: torch.Tensor) -> int:
    return int(weight.numel()) * 2


def fit_lowrank_overlay(
    residual: torch.Tensor,
    *,
    rank: int,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    min_dim = min(int(residual.shape[0]), int(residual.shape[1]))
    if min_dim < 1:
        raise ValueError("residual must be at least 1x1")
    eff_rank = min(max(int(rank), 1), min_dim)
    q = min(min_dim, max(eff_rank, eff_rank + max(int(oversample), 0)))
    u, s, v = torch.svd_lowrank(residual.to(torch.float32), q=q, niter=max(int(niter), 0))
    left = (u[:, :eff_rank] * s[:eff_rank]).to(torch.float16).contiguous()
    right = v[:, :eff_rank].transpose(0, 1).to(torch.float16).contiguous()
    return left, right, q


def overlay_storage_bytes(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(left.numel() + right.numel()) * 2


def probe_one(
    name: str,
    *,
    device: str,
    ranks: list[int],
    group_rows: int,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    stages_per_split: tuple[int, ...] | None,
    normalize_blocks: str,
    transform_kind: str,
    sample_limit: int,
    kmeans_iters: int,
    batch_size: int,
    svd_oversample: int,
    svd_niter: int,
) -> dict[str, object]:
    print(f"\n=== {name} ===", flush=True)
    weight = load_weight(name, device)
    pq_enc = encode_grouped_block_residual_vq(
        weight,
        group_rows=group_rows,
        block_size=block_size,
        codebook_size=codebook_size,
        num_stages=num_stages,
        product_splits=product_splits,
        stages_per_split=stages_per_split,
        normalize_blocks=normalize_blocks,
        transform_kind=transform_kind,
        sample_limit=sample_limit,
        kmeans_iters=kmeans_iters,
        batch_size=batch_size,
    )
    pq_recon = pq_enc.reconstruct(out_dtype=torch.float32)
    residual = weight.float() - pq_recon
    pq_storage = int(pq_enc.storage_bytes())
    pq_rel = rel_mse(pq_recon, weight)
    pq_bits = float(pq_enc.bits_per_weight())
    pq_sample_rel = float(pq_enc.sample_rel_mse)
    residual_energy = float(residual.square().mean().item())
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    dense_mb = storage_megabytes(dense_bf16_storage_bytes(weight))
    print(
        f"  pq_only: rel_mse={pq_rel:.3e} storage_mb={storage_megabytes(pq_storage):.2f} bits/weight={pq_bits:.3f}",
        flush=True,
    )

    rank_results = []
    for rank in ranks:
        left, right, svd_q = fit_lowrank_overlay(
            residual,
            rank=rank,
            oversample=svd_oversample,
            niter=svd_niter,
        )
        overlay = left.float() @ right.float()
        hybrid_recon = pq_recon + overlay
        hybrid_rel = rel_mse(hybrid_recon, weight)
        remaining = residual - overlay
        remaining_energy = float(remaining.square().mean().item())
        capture = 1.0 - (remaining_energy / max(residual_energy, 1e-12))
        extra_storage = overlay_storage_bytes(left, right)
        total_storage = pq_storage + extra_storage
        total_bits = 8.0 * float(total_storage) / max(rows * cols, 1)
        improvement = pq_rel / max(hybrid_rel, 1e-12)
        rank_results.append(
            {
                "rank": int(rank),
                "svd_q": int(svd_q),
                "hybrid_rel_mse": hybrid_rel,
                "improvement_vs_pq": improvement,
                "residual_capture": capture,
                "overlay_storage_mb": storage_megabytes(extra_storage),
                "total_storage_mb": storage_megabytes(total_storage),
                "total_bits_per_weight": total_bits,
            }
        )
        print(
            f"  +rank{rank:>2d}: rel_mse={hybrid_rel:.3e} improve={improvement:.2f}x residual_capture={capture:.3f} total_mb={storage_megabytes(total_storage):.2f}",
            flush=True,
        )

    del pq_recon, residual, pq_enc, weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "shape": [rows, cols],
        "dense_bf16_mb": dense_mb,
        "pq_only": {
            "rel_mse": pq_rel,
            "storage_mb": storage_megabytes(pq_storage),
            "bits_per_weight": pq_bits,
            "sample_rel_mse": pq_sample_rel,
        },
        "overlay_ranks": rank_results,
    }


def parse_names(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return DEFAULT_TENSOR_NAMES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_ranks(raw: str) -> list[int]:
    ranks = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ranks:
        raise ValueError("at least one rank is required")
    return ranks


def parse_stage_schedule(raw: str, product_splits: int) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    schedule = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(schedule) != product_splits:
        raise ValueError("stages-per-split must have exactly product_splits entries")
    if any(stage < 1 for stage in schedule):
        raise ValueError("all stages-per-split values must be >= 1")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-names", default=",".join(DEFAULT_TENSOR_NAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="2,4,8,16")
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--stages-per-split", default="")
    parser.add_argument("--normalize-blocks", choices=("none", "amax", "l2"), default="none")
    parser.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "pca"), default="none")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--svd-oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(ROOT / "dwl2_dynamic_route/results/m12a_pq_lowrank_overlay_probe.json"),
    )
    args = parser.parse_args()

    tensor_names = parse_names(args.tensor_names)
    ranks = parse_ranks(args.ranks)
    stages_per_split = parse_stage_schedule(args.stages_per_split, args.product_splits)

    results = {
        "tensor_names": list(tensor_names),
        "ranks": ranks,
        "pq_cfg": {
            "group_rows": args.group_rows,
            "block_size": args.block_size,
            "codebook_size": args.codebook_size,
            "num_stages": args.num_stages,
            "product_splits": args.product_splits,
            "stages_per_split": None if stages_per_split is None else list(stages_per_split),
            "normalize_blocks": args.normalize_blocks,
            "transform_kind": args.transform_kind,
            "sample_limit": args.sample_limit,
            "kmeans_iters": args.kmeans_iters,
            "batch_size": args.batch_size,
        },
        "svd_cfg": {
            "oversample": args.svd_oversample,
            "niter": args.svd_niter,
        },
        "layers": [],
    }

    for name in tensor_names:
        try:
            results["layers"].append(
                probe_one(
                    name,
                    device=args.device,
                    ranks=ranks,
                    group_rows=args.group_rows,
                    block_size=args.block_size,
                    codebook_size=args.codebook_size,
                    num_stages=args.num_stages,
                    product_splits=args.product_splits,
                    stages_per_split=stages_per_split,
                    normalize_blocks=args.normalize_blocks,
                    transform_kind=args.transform_kind,
                    sample_limit=args.sample_limit,
                    kmeans_iters=args.kmeans_iters,
                    batch_size=args.batch_size,
                    svd_oversample=args.svd_oversample,
                    svd_niter=args.svd_niter,
                )
            )
        except Exception as exc:
            print(f"  FAILED {name}: {exc}", flush=True)
            results["layers"].append({"name": name, "error": str(exc)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    print("\n=== SUMMARY ===")
    for item in results["layers"]:
        if "error" in item:
            print(f"  {item['name']}: ERROR {item['error']}")
            continue
        best = min(item["overlay_ranks"], key=lambda row: row["hybrid_rel_mse"])
        print(
            f"  {item['name']:45s} pq_rel_mse={item['pq_only']['rel_mse']:.3e} best_rank={best['rank']:>2d} best_rel_mse={best['hybrid_rel_mse']:.3e} total_mb={best['total_storage_mb']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()