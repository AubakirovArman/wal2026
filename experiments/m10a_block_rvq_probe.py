"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from dwl2_dynamic_route.src.block_vq import (
    dense_bf16_storage_bytes,
    encode_block_residual_vq,
    encode_grouped_block_residual_vq,
    sample_grouped_row_similarity,
    sample_row_similarity,
    storage_megabytes,
)
from dwl2_dynamic_route.src.runtime import PackedIDRouteLinear, quantize_linear_to_packed


MODEL_DIR = Path(
    "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
    "models--unsloth--Llama-3.3-70B-Instruct/snapshots/"
    "99cd0d2c829e92a67c844f9144c2509632e5c87f"
)
DEVICE = "cuda:0"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]


def load_weight(name: str) -> torch.Tensor:
    shard = MODEL_DIR / INDEX[name]
    with safe_open(str(shard), framework="pt", device=DEVICE) as handle:
        return handle.get_tensor(name).to(torch.bfloat16)


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).square().mean()
    den = b.float().square().mean().clamp_min(1e-12)
    return float((diff / den).item())


def route_runtime_storage_bytes(packed: PackedIDRouteLinear) -> int:
    total = 0
    total += int(packed.ids.numel()) * packed.ids.element_size()
    total += int(packed.codebook_sum.numel()) * packed.codebook_sum.element_size()
    total += int(packed.row_scale.numel()) * packed.row_scale.element_size()
    if packed.bias is not None:
        total += int(packed.bias.numel()) * packed.bias.element_size()
    return total


def probe_layer(name: str, configs: list[dict[str, int]]) -> dict:
    print(f"\n=== {name} ===", flush=True)
    weight = load_weight(name)
    print(f"  weight shape={tuple(weight.shape)} dtype={weight.dtype}", flush=True)

    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    t0 = time.time()
    packed, route_stats = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    route_recon = packed.reconstruct_weight().to(torch.float32)
    route_sim = sample_row_similarity((packed.ids.to(torch.int32),))
    route_storage = route_runtime_storage_bytes(packed)
    print(
        f"  route: rel_mse={route_stats['rel_mse']:.3e}  "
        f"runtime_mb={storage_megabytes(route_storage):.2f}  "
        f"row_sim={route_sim['mean_sim']:.4f}",
        flush=True,
    )

    result = {
        "name": name,
        "shape": list(weight.shape),
        "bf16_mb": storage_megabytes(dense_bf16_storage_bytes(weight)),
        "route": {
            "rel_mse": route_stats["rel_mse"],
            "runtime_mb": storage_megabytes(route_storage),
            "unique_routes": route_stats["unique_routes"],
            "row_similarity": route_sim,
        },
        "block_rvq": [],
    }

    for cfg in configs:
        t1 = time.time()
        if "group_rows" in cfg:
            enc = encode_grouped_block_residual_vq(
                weight,
                group_rows=cfg["group_rows"],
                block_size=cfg["block_size"],
                codebook_size=cfg["codebook_size"],
                num_stages=cfg["num_stages"],
                normalize_blocks=cfg.get("normalize_blocks", "none"),
                sample_limit=cfg["sample_limit"],
                kmeans_iters=cfg["kmeans_iters"],
                batch_size=cfg["batch_size"],
            )
            sim = sample_grouped_row_similarity(enc.groups)
            label = f"grouped_rvq g={cfg['group_rows']:>4d} b={cfg['block_size']:>2d} c={cfg['codebook_size']:>4d} s={cfg['num_stages']} n={cfg.get('normalize_blocks', 'none')}"
        else:
            enc = encode_block_residual_vq(
                weight,
                block_size=cfg["block_size"],
                codebook_size=cfg["codebook_size"],
                num_stages=cfg["num_stages"],
                normalize_blocks=cfg.get("normalize_blocks", "none"),
                sample_limit=cfg["sample_limit"],
                kmeans_iters=cfg["kmeans_iters"],
                batch_size=cfg["batch_size"],
            )
            sim = sample_row_similarity(enc.stage_ids)
            label = f"block_rvq   b={cfg['block_size']:>2d} c={cfg['codebook_size']:>4d} s={cfg['num_stages']} n={cfg.get('normalize_blocks', 'none')}"
        recon = enc.reconstruct(out_dtype=torch.float32)
        err = rel_mse(recon, weight)
        runtime_mb = storage_megabytes(enc.storage_bytes())
        route_ratio = runtime_mb / max(storage_megabytes(route_storage), 1e-9)
        dt = time.time() - t1
        print(
            f"  {label}: "
            f"rel_mse={err:.3e}  runtime_mb={runtime_mb:.2f}  vs_route={route_ratio:.3f}x  "
            f"row_sim={sim['mean_sim']:.4f}  sample_rel_mse={enc.sample_rel_mse:.3e}  ({dt:.1f}s)",
            flush=True,
        )
        result["block_rvq"].append(
            {
                "config": cfg,
                "rel_mse": err,
                "runtime_mb": runtime_mb,
                "runtime_vs_route_ratio": route_ratio,
                "row_similarity": sim,
                "sample_rel_mse": enc.sample_rel_mse,
            }
        )
        del recon, enc
        torch.cuda.empty_cache()

    del packed, route_recon, linear, weight
    torch.cuda.empty_cache()
    return result


def main() -> None:
    configs = [
        {"block_size": 32, "codebook_size": 256, "num_stages": 2, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"block_size": 32, "codebook_size": 256, "num_stages": 4, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"block_size": 16, "codebook_size": 256, "num_stages": 2, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"block_size": 32, "codebook_size": 512, "num_stages": 2, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"group_rows": 2048, "block_size": 32, "codebook_size": 256, "num_stages": 2, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"group_rows": 2048, "block_size": 32, "codebook_size": 256, "num_stages": 4, "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"block_size": 32, "codebook_size": 256, "num_stages": 2, "normalize_blocks": "amax", "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"block_size": 32, "codebook_size": 256, "num_stages": 4, "normalize_blocks": "amax", "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"group_rows": 2048, "block_size": 32, "codebook_size": 256, "num_stages": 2, "normalize_blocks": "amax", "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
        {"group_rows": 2048, "block_size": 32, "codebook_size": 256, "num_stages": 4, "normalize_blocks": "amax", "sample_limit": 65_536, "kmeans_iters": 8, "batch_size": 16_384},
    ]
    layers = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.40.mlp.gate_proj.weight",
    ]

    results = []
    for name in layers:
        try:
            results.append(probe_layer(name, configs))
        except Exception as exc:
            print(f"  FAILED {name}: {exc}", flush=True)
            results.append({"name": name, "error": str(exc)})

    out_path = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m10a_block_rvq_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    print("\n=== SUMMARY ===")
    for item in results:
        if "error" in item:
            print(f"  {item['name']}: ERROR")
            continue
        best = min(item["block_rvq"], key=lambda row: row["rel_mse"])
        print(
            f"  {item['name']:55s} route_rel_mse={item['route']['rel_mse']:.3e} "
            f"best_block_rel_mse={best['rel_mse']:.3e} "
            f"best_block_runtime_vs_route={best['runtime_vs_route_ratio']:.3f}x "
            f"best_block_row_sim={best['row_similarity']['mean_sim']:.4f}"
        )


if __name__ == "__main__":
    main()