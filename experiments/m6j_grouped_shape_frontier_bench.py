from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.codebook import build_codebook
from dwl2_dynamic_route.src.full_layer_tiled_runtime import (
    build_grouped_local_plan,
    full_layer_grouped_global_matmul,
    full_layer_grouped_local_matmul,
    launch_count,
    mean_local_unique,
    total_local_unique,
)
from dwl2_dynamic_route.src.route_encoder import encode_routes
from dwl2_dynamic_route.src.triton_id_matmul import id_route_linear_matmul


def _load_weight(tensor_name: str, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(MODEL_DIR / "model.safetensors.index.json") as handle:
        shard_map = json.load(handle)["weight_map"]
    shard_path = MODEL_DIR / shard_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _bench_cuda(fn, reps: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _targets(policy_json: str) -> list[str]:
    with open(policy_json) as handle:
        policy = json.load(handle)
    return [item["tensor_name"] for item in policy["approved"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-json", default=str(ROOT / "dwl2_dynamic_route/results/m6f_selective_runtime_policy.json"))
    parser.add_argument("--group-rows", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--group-cols", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--bench-tokens", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6j_grouped_shape_frontier_bench.json"))
    args = parser.parse_args()

    targets = _targets(args.policy_json)
    if not targets:
        raise ValueError("no approved tensors found in policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict[str, object]] = []
    for tensor_name in targets:
        weight = _load_weight(tensor_name, device=device)
        row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = weight / row_scale
        sample = w_norm.flatten()
        if sample.numel() > args.sample_limit:
            idx = torch.randint(0, sample.numel(), (args.sample_limit,), device=sample.device)
            sample = sample[idx]
        ladder = calibrate_ladder(sample, l_max=12, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric")
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=12)
        codebook, ids = build_codebook(enc.digits, enc.stop_depth, l_max=12)
        codebook_sum = (codebook.digits.to(torch.float32) * ladder.to(torch.float32)).sum(dim=-1)
        routed_norm = codebook_sum[ids.long()]
        dense_w = (routed_norm * row_scale).to(torch.float16).contiguous()
        x = torch.randn(args.bench_tokens, weight.shape[1], device=device, dtype=torch.float16)
        dense_ref = F.linear(x, dense_w)
        ms_dense = _bench_cuda(lambda: F.linear(x, dense_w))
        ms_global_full = _bench_cuda(lambda: id_route_linear_matmul(x, ids, codebook_sum, row_scale))
        for group_rows in args.group_rows:
            capped_rows = min(group_rows, weight.shape[0])
            for group_cols in args.group_cols:
                capped_cols = min(group_cols, weight.shape[1])
                plan = build_grouped_local_plan(routed_norm, capped_rows, capped_cols)
                local_ref = full_layer_grouped_local_matmul(x, plan, row_scale)
                global_ref = full_layer_grouped_global_matmul(x, ids, codebook_sum, row_scale, capped_rows, capped_cols)
                ms_local = _bench_cuda(lambda: full_layer_grouped_local_matmul(x, plan, row_scale))
                ms_global = _bench_cuda(lambda: full_layer_grouped_global_matmul(x, ids, codebook_sum, row_scale, capped_rows, capped_cols))
                total_unique = total_local_unique(plan)
                launches = launch_count(plan)
                results.append(
                    {
                        "tensor_name": tensor_name,
                        "group_rows": int(capped_rows),
                        "group_cols": int(capped_cols),
                        "launches": launches,
                        "mean_group_unique": mean_local_unique(plan),
                        "total_group_unique": total_unique,
                        "group_area": int(capped_rows * capped_cols),
                        "ms_dense": ms_dense,
                        "ms_global_full": ms_global_full,
                        "ms_grouped_global": ms_global,
                        "ms_grouped_local": ms_local,
                        "local_vs_grouped_global": ms_global / max(ms_local, 1e-12),
                        "local_vs_global_full": ms_global_full / max(ms_local, 1e-12),
                        "local_vs_dense": ms_dense / max(ms_local, 1e-12),
                        "grouped_global_mse": float(torch.mean((global_ref - dense_ref).float().square()).item()),
                        "grouped_local_mse": float(torch.mean((local_ref - dense_ref).float().square()).item()),
                    }
                )
                print(
                    f"[m6j] {tensor_name} gr={capped_rows:>5} gc={capped_cols:>5} launches={launches:>5} "
                    f"mean_unique={mean_local_unique(plan):>8.1f} total_unique={total_unique:>8} "
                    f"local={ms_local:>8.3f} ms global={ms_global:>8.3f} ms"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"[m6j] wrote {out_path}")


if __name__ == "__main__":
    main()