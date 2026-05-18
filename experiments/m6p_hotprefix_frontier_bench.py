from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.codebook import build_codebook
from dwl2_dynamic_route.src.full_layer_tiled_runtime import build_grouped_local_plan, full_layer_grouped_global_matmul, full_layer_grouped_local_matmul, launch_count
from dwl2_dynamic_route.src.route_encoder import encode_routes
from dwl2_dynamic_route.src.triton_id_matmul import id_route_linear_matmul
from dwl2_dynamic_route.src.triton_local_palette_hotprefix_matmul import local_palette_hotprefix_linear_matmul


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


def _full_layer_grouped_local_hotprefix(x, plan, row_scale, hot_size):
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_tiles = {(int(item["col0"]), int(item["col1"])): x_2d[:, int(item["col0"]):int(item["col1"])].contiguous() for item in plan}
    out_features = max(int(item["row1"]) for item in plan)
    out = torch.zeros((x_2d.shape[0], out_features), device=x.device, dtype=torch.float32)
    for item in plan:
        row0 = int(item["row0"])
        row1 = int(item["row1"])
        col0 = int(item["col0"])
        col1 = int(item["col1"])
        out[:, row0:row1] += local_palette_hotprefix_linear_matmul(
            x_tiles[(col0, col1)], item["local_idx"], item["palette"], row_scale[row0:row1], hot_size=hot_size
        ).float()
    return out.to(x.dtype).reshape(*x.shape[:-1], out_features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-json", default=str(ROOT / "dwl2_dynamic_route/results/m6f_selective_runtime_policy.json"))
    parser.add_argument("--group-rows", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--group-cols", type=int, nargs="+", default=[8192])
    parser.add_argument("--bench-tokens", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--hot-size", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6p_hotprefix_frontier_bench.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for tensor_name in _targets(args.policy_json):
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
        ms_global_full = _bench_cuda(lambda: id_route_linear_matmul(x, ids, codebook_sum, row_scale))
        for group_rows in args.group_rows:
            for group_cols in args.group_cols:
                plan = build_grouped_local_plan(routed_norm, min(group_rows, weight.shape[0]), min(group_cols, weight.shape[1]))
                local_ref = full_layer_grouped_local_matmul(x, plan, row_scale)
                hot_ref = _full_layer_grouped_local_hotprefix(x, plan, row_scale, args.hot_size)
                global_ref = full_layer_grouped_global_matmul(x, ids, codebook_sum, row_scale, min(group_rows, weight.shape[0]), min(group_cols, weight.shape[1]))
                ms_local = _bench_cuda(lambda: full_layer_grouped_local_matmul(x, plan, row_scale))
                ms_hot = _bench_cuda(lambda: _full_layer_grouped_local_hotprefix(x, plan, row_scale, args.hot_size))
                ms_global = _bench_cuda(lambda: full_layer_grouped_global_matmul(x, ids, codebook_sum, row_scale, min(group_rows, weight.shape[0]), min(group_cols, weight.shape[1])))
                results.append({
                    "tensor_name": tensor_name,
                    "group_rows": int(min(group_rows, weight.shape[0])),
                    "group_cols": int(min(group_cols, weight.shape[1])),
                    "hot_size": int(args.hot_size),
                    "launches": launch_count(plan),
                    "ms_global_full": ms_global_full,
                    "ms_grouped_global": ms_global,
                    "ms_grouped_local": ms_local,
                    "ms_grouped_local_hot": ms_hot,
                    "hot_vs_local": ms_local / max(ms_hot, 1e-12),
                    "hot_vs_grouped_global": ms_global / max(ms_hot, 1e-12),
                    "hot_vs_global_full": ms_global_full / max(ms_hot, 1e-12),
                    "grouped_local_mse": float(torch.mean((local_ref - dense_ref).float().square()).item()),
                    "grouped_local_hot_mse": float(torch.mean((hot_ref - dense_ref).float().square()).item()),
                    "grouped_global_mse": float(torch.mean((global_ref - dense_ref).float().square()).item()),
                })
                print(
                    f"[m6p] {tensor_name} gr={group_rows:>4} gc={group_cols:>4} hot={args.hot_size:>2} "
                    f"local={ms_local:>7.3f} hot={ms_hot:>7.3f} speedup={ms_local / max(ms_hot, 1e-12):.3f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"[m6p] wrote {out_path}")


if __name__ == "__main__":
    main()