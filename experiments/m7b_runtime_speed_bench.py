"""Runtime microbench for ID-route and Block-RVQ layer variants.

Supports:
  * id_route        - existing packed/fused/cached route benchmark
  * block_rvq       - single-layer Block-RVQ packed benchmark
  * block_rvq_bundle - q/k/v/o upper-bound bench for shared decode ideas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import encode_grouped_block_residual_vq
from dwl2_dynamic_route.src.runtime import (
    CachedPackedIDRouteLinear,
    FusedIDRouteLinear,
    PackedGroupedBlockRVQLinear,
    PackedIDRouteLinear,
    quantize_linear_to_packed,
)

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
DEFAULT_CONFIGS = [(1, 1), (1, 32), (1, 512), (1, 2048), (4, 2048)]
DEFAULT_ID_ROUTE_TENSORS = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]
DEFAULT_BLOCK_RVQ_TENSORS = [
    "model.language_model.layers.54.self_attn.q_proj.weight",
    "model.language_model.layers.54.self_attn.k_proj.weight",
]


def load_weight(name: str, device: str = "cuda:3") -> torch.Tensor:
    shard = MODEL_DIR / INDEX[name]
    with safe_open(shard, framework="pt", device=device) as handle:
        return handle.get_tensor(name).to(torch.bfloat16)


def parse_configs(text: str) -> list[tuple[int, int]]:
    if not text:
        return list(DEFAULT_CONFIGS)
    configs = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        bs_text, seq_text = item.split("x", 1)
        configs.append((int(bs_text), int(seq_text)))
    return configs


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_call(fn, *args, warmup: int = 3, iters: int = 10):
    for _ in range(warmup):
        fn(*args)
    sync_if_needed()
    t0 = time.time()
    result = None
    for _ in range(iters):
        result = fn(*args)
    sync_if_needed()
    return (time.time() - t0) / max(iters, 1), result


def build_linear(weight: torch.Tensor, device: str) -> nn.Linear:
    out_features, in_features = weight.shape
    linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


def block_rvq_rel_mse(weight: torch.Tensor, packed: PackedGroupedBlockRVQLinear) -> float:
    recon = packed.reconstruct_weight(out_dtype=torch.float32)
    ref = weight.to(torch.float32)
    return float(((ref - recon).square().mean() / ref.square().mean().clamp_min(1e-12)).item())


def bench_id_route_layer(tensor_name: str, configs: list[tuple[int, int]], *, device: str, warmup: int, iters: int):
    weight = load_weight(tensor_name, device)
    linear = build_linear(weight, device)
    bf16_weight = linear.weight

    packed, _ = quantize_linear_to_packed(linear, runtime_cls=PackedIDRouteLinear)
    packed = packed.to(device)
    fused, _ = quantize_linear_to_packed(linear, runtime_cls=FusedIDRouteLinear)
    fused = fused.to(device)
    cached, _ = quantize_linear_to_packed(
        linear,
        runtime_cls=CachedPackedIDRouteLinear,
        runtime_kwargs={"max_cache_bytes": 512 * 2**20},
    )
    cached = cached.to(device)

    for _ in range(2):
        cached(torch.randn(1, 32, weight.shape[1], device=device, dtype=torch.bfloat16))

    rows = []
    for bs, seq in configs:
        x = torch.randn(bs, seq, weight.shape[1], device=device, dtype=torch.bfloat16)
        t_bf16, _ = bench_call(lambda x_in: F.linear(x_in, bf16_weight), x, warmup=warmup, iters=iters)
        t_packed, _ = bench_call(packed, x, warmup=warmup, iters=iters)
        t_fused, _ = bench_call(fused, x, warmup=warmup, iters=iters)
        t_cached, _ = bench_call(cached, x, warmup=warmup, iters=iters)
        row = {
            "mode": "id_route",
            "tensor": tensor_name,
            "bs": bs,
            "seq": seq,
            "bf16_ms": round(t_bf16 * 1000.0, 3),
            "packed_ms": round(t_packed * 1000.0, 3),
            "fused_ms": round(t_fused * 1000.0, 3),
            "cached_ms": round(t_cached * 1000.0, 3),
            "packed_vs_bf16": round(t_packed / t_bf16, 3),
            "fused_vs_bf16": round(t_fused / t_bf16, 3),
            "cached_vs_bf16": round(t_cached / t_bf16, 3),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    del packed, fused, cached, linear, weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_block_rvq_layer(
    weight: torch.Tensor,
    *,
    group_rows: int,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    transform_kind: str,
    calibrate_stage_scales: bool,
    residual_correction: str,
    sample_limit: int,
    kmeans_iters: int,
    batch_size: int,
    matmul_strategy: str,
) -> tuple[PackedGroupedBlockRVQLinear, dict[str, float | int | str | bool]]:
    enc = encode_grouped_block_residual_vq(
        weight.to(torch.float32),
        group_rows=group_rows,
        block_size=block_size,
        codebook_size=codebook_size,
        num_stages=num_stages,
        product_splits=product_splits,
        normalize_blocks="none",
        transform_kind=transform_kind,
        calibrate_stage_scales=calibrate_stage_scales,
        residual_correction=residual_correction,
        sample_limit=sample_limit,
        kmeans_iters=kmeans_iters,
        batch_size=batch_size,
    )
    packed = PackedGroupedBlockRVQLinear(enc, bias=None, matmul_strategy=matmul_strategy)
    stats = {
        "storage_bytes": int(enc.storage_bytes()),
        "bits_per_weight": float(enc.bits_per_weight()),
        "sample_rel_mse": float(enc.sample_rel_mse),
        "groups": int(len(enc.groups)),
    }
    return packed, stats


def bench_block_rvq_layer(
    tensor_name: str,
    configs: list[tuple[int, int]],
    *,
    device: str,
    warmup: int,
    iters: int,
    group_rows: int,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    transform_kind: str,
    calibrate_stage_scales: bool,
    residual_correction: str,
    sample_limit: int,
    kmeans_iters: int,
    batch_size: int,
    matmul_strategy: str,
):
    weight = load_weight(tensor_name, device)
    linear = build_linear(weight, device)
    packed, stats = build_block_rvq_layer(
        weight,
        group_rows=group_rows,
        block_size=block_size,
        codebook_size=codebook_size,
        num_stages=num_stages,
        product_splits=product_splits,
        transform_kind=transform_kind,
        calibrate_stage_scales=calibrate_stage_scales,
        residual_correction=residual_correction,
        sample_limit=sample_limit,
        kmeans_iters=kmeans_iters,
        batch_size=batch_size,
        matmul_strategy=matmul_strategy,
    )
    packed = packed.to(device)
    rel_mse = block_rvq_rel_mse(weight, packed)

    rows = []
    for bs, seq in configs:
        x = torch.randn(bs, seq, weight.shape[1], device=device, dtype=torch.bfloat16)
        t_bf16, _ = bench_call(linear, x, warmup=warmup, iters=iters)
        t_packed, _ = bench_call(packed, x, warmup=warmup, iters=iters)
        row = {
            "mode": "block_rvq",
            "tensor": tensor_name,
            "bs": bs,
            "seq": seq,
            "bf16_ms": round(t_bf16 * 1000.0, 3),
            "packed_ms": round(t_packed * 1000.0, 3),
            "packed_vs_bf16": round(t_packed / t_bf16, 3),
            "rel_mse": round(rel_mse, 8),
            "group_rows": group_rows,
            "block_size": block_size,
            "codebook_size": codebook_size,
            "num_stages": num_stages,
            "product_splits": product_splits,
            "transform_kind": transform_kind,
            "calibrate_stage_scales": calibrate_stage_scales,
            "residual_correction": residual_correction,
            "matmul_strategy": matmul_strategy,
            **stats,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    del packed, linear, weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def bench_block_rvq_bundle(
    layer_idx: int,
    configs: list[tuple[int, int]],
    *,
    device: str,
    warmup: int,
    iters: int,
    group_rows: int,
    block_size: int,
    codebook_size: int,
    num_stages: int,
    product_splits: int,
    transform_kind: str,
    calibrate_stage_scales: bool,
    residual_correction: str,
    sample_limit: int,
    kmeans_iters: int,
    batch_size: int,
    matmul_strategy: str,
):
    tensor_names = [
        f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight",
        f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight",
    ]
    weights = [load_weight(name, device) for name in tensor_names]
    packed_layers = []
    for weight in weights:
        layer, _ = build_block_rvq_layer(
            weight,
            group_rows=group_rows,
            block_size=block_size,
            codebook_size=codebook_size,
            num_stages=num_stages,
            product_splits=product_splits,
            transform_kind=transform_kind,
            calibrate_stage_scales=calibrate_stage_scales,
            residual_correction=residual_correction,
            sample_limit=sample_limit,
            kmeans_iters=kmeans_iters,
            batch_size=batch_size,
            matmul_strategy=matmul_strategy,
        )
        packed_layers.append(layer.to(device))

    bucket_specs: list[tuple[int, list[PackedGroupedBlockRVQLinear]]] = []
    bucket_map: dict[int, list[PackedGroupedBlockRVQLinear]] = {}
    for layer in packed_layers:
        bucket_map.setdefault(layer.out_features, []).append(layer)
    for out_features, layers in sorted(bucket_map.items()):
        bucket_specs.append((out_features, layers))

    def reconstruct_all(dtype: torch.dtype) -> list[tuple[int, torch.Tensor]]:
        return [
            (
                out_features,
                torch.stack([layer.reconstruct_weight(out_dtype=dtype) for layer in layers], dim=0),
            )
            for out_features, layers in bucket_specs
        ]

    def run_individual(x: torch.Tensor):
        return [layer(x) for layer in packed_layers]

    def run_bundle_upper(x: torch.Tensor):
        x_flat = x.reshape(-1, x.shape[-1])
        outputs = []
        for out_features, weights_stack in reconstruct_all(x.dtype):
            out = torch.matmul(x_flat.unsqueeze(0), weights_stack.transpose(1, 2).contiguous())
            outputs.append(out.view(weights_stack.shape[0], *x.shape[:-1], out_features))
        return outputs

    rows = []
    mean_rel_mse = float(
        sum(block_rvq_rel_mse(weight, layer) for weight, layer in zip(weights, packed_layers)) / max(len(packed_layers), 1)
    )
    for bs, seq in configs:
        x = torch.randn(bs, seq, weights[0].shape[1], device=device, dtype=torch.bfloat16)
        t_individual, _ = bench_call(run_individual, x, warmup=warmup, iters=iters)
        t_reconstruct, weights_stack = bench_call(lambda dtype: reconstruct_all(dtype), x.dtype, warmup=warmup, iters=iters)
        t_matmul_only, _ = bench_call(
            lambda x_in, stacks: [
                torch.matmul(x_in.reshape(-1, x_in.shape[-1]).unsqueeze(0), stack.transpose(1, 2).contiguous())
                for _, stack in stacks
            ],
            x,
            weights_stack,
            warmup=warmup,
            iters=iters,
        )
        t_bundle_upper, _ = bench_call(run_bundle_upper, x, warmup=warmup, iters=iters)
        row = {
            "mode": "block_rvq_bundle",
            "layer_idx": layer_idx,
            "bs": bs,
            "seq": seq,
            "individual_ms": round(t_individual * 1000.0, 3),
            "reconstruct_all_ms": round(t_reconstruct * 1000.0, 3),
            "stacked_matmul_only_ms": round(t_matmul_only * 1000.0, 3),
            "bundle_upper_ms": round(t_bundle_upper * 1000.0, 3),
            "bundle_vs_individual": round(t_bundle_upper / t_individual, 3),
            "mean_rel_mse": round(mean_rel_mse, 8),
            "bundle_buckets": ",".join(f"{stack.shape[0]}x{out_features}" for out_features, stack in weights_stack),
            "group_rows": group_rows,
            "transform_kind": transform_kind,
            "calibrate_stage_scales": calibrate_stage_scales,
            "residual_correction": residual_correction,
            "matmul_strategy": matmul_strategy,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)

    del packed_layers, weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("id_route", "block_rvq", "block_rvq_bundle"), default="block_rvq")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--configs", default="1x1,1x32,1x512,1x2048,4x2048")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--tensors", default=",".join(DEFAULT_BLOCK_RVQ_TENSORS))
    parser.add_argument("--layer-idx", type=int, default=54)
    parser.add_argument("--group-rows", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default="none")
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--residual-correction", choices=("none", "sign"), default="none")
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--matmul-strategy",
        choices=("per_group", "full_weight", "chunked_weight", "local_palette", "stagewise_einsum", "stacked_matmul"),
        default="per_group",
    )
    parser.add_argument("--matmul-chunk-rows", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m7b_runtime_speed_bench.json"))
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    rows = []
    if args.mode == "id_route":
        tensors = [item.strip() for item in args.tensors.split(",") if item.strip()]
        if not tensors:
            tensors = list(DEFAULT_ID_ROUTE_TENSORS)
        for tensor_name in tensors:
            print(f"\n=== {tensor_name} ===", flush=True)
            rows.extend(bench_id_route_layer(tensor_name, configs, device=args.device, warmup=args.warmup, iters=args.iters))
    elif args.mode == "block_rvq":
        tensors = [item.strip() for item in args.tensors.split(",") if item.strip()]
        if not tensors:
            tensors = list(DEFAULT_BLOCK_RVQ_TENSORS)
        for tensor_name in tensors:
            print(f"\n=== {tensor_name} ===", flush=True)
            rows.extend(
                bench_block_rvq_layer(
                    tensor_name,
                    configs,
                    device=args.device,
                    warmup=args.warmup,
                    iters=args.iters,
                    group_rows=args.group_rows,
                    block_size=args.block_size,
                    codebook_size=args.codebook_size,
                    num_stages=args.num_stages,
                    product_splits=args.product_splits,
                    transform_kind=args.transform_kind,
                    calibrate_stage_scales=args.calibrate_stage_scales,
                    residual_correction=args.residual_correction,
                    sample_limit=args.sample_limit,
                    kmeans_iters=args.kmeans_iters,
                    batch_size=args.batch_size,
                    matmul_strategy=args.matmul_strategy,
                )
            )
    else:
        print(f"\n=== layer {args.layer_idx} qkvo bundle ===", flush=True)
        rows.extend(
            bench_block_rvq_bundle(
                args.layer_idx,
                configs,
                device=args.device,
                warmup=args.warmup,
                iters=args.iters,
                group_rows=args.group_rows,
                block_size=args.block_size,
                codebook_size=args.codebook_size,
                num_stages=args.num_stages,
                product_splits=args.product_splits,
                transform_kind=args.transform_kind,
                calibrate_stage_scales=args.calibrate_stage_scales,
                residual_correction=args.residual_correction,
                sample_limit=args.sample_limit,
                kmeans_iters=args.kmeans_iters,
                batch_size=args.batch_size,
                matmul_strategy=args.matmul_strategy,
            )
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
