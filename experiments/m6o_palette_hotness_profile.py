from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.codebook import build_codebook
from dwl2_dynamic_route.src.full_layer_tiled_runtime import build_grouped_local_plan
from dwl2_dynamic_route.src.route_encoder import encode_routes


def _load_weight(tensor_name: str, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(MODEL_DIR / "model.safetensors.index.json") as handle:
        shard_map = json.load(handle)["weight_map"]
    shard_path = MODEL_DIR / shard_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device=device) as handle:
        return handle.get_tensor(tensor_name)


def _targets(policy_json: str) -> list[str]:
    with open(policy_json) as handle:
        policy = json.load(handle)
    return [item["tensor_name"] for item in policy["approved"]]


def _coverage(counts: torch.Tensor, topk: int) -> float:
    use_k = min(topk, counts.numel())
    if use_k == 0:
        return 0.0
    top_sum = torch.topk(counts, k=use_k, largest=True).values.sum()
    return float((top_sum / counts.sum().clamp_min(1)).item())


def _min_k_for_threshold(counts: torch.Tensor, threshold: float) -> int:
    sorted_counts = torch.sort(counts, descending=True).values
    cdf = torch.cumsum(sorted_counts, dim=0).to(torch.float32) / counts.sum().clamp_min(1)
    reached = torch.nonzero(cdf >= threshold, as_tuple=False)
    return int(reached[0].item() + 1) if reached.numel() else int(sorted_counts.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-json", default=str(ROOT / "dwl2_dynamic_route/results/m6f_selective_runtime_policy.json"))
    parser.add_argument("--group-rows", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--group-cols", type=int, nargs="+", default=[8192])
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6o_palette_hotness_profile.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict[str, object]] = []
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
        routed_norm = (codebook.digits.to(torch.float32) * ladder.to(torch.float32)).sum(dim=-1)[ids.long()]
        for group_rows in args.group_rows:
            capped_rows = min(group_rows, weight.shape[0])
            for group_cols in args.group_cols:
                capped_cols = min(group_cols, weight.shape[1])
                plan = build_grouped_local_plan(routed_norm, capped_rows, capped_cols)
                group_stats: list[dict[str, float | int]] = []
                for item in plan:
                    local_idx = item["local_idx"].reshape(-1).to(torch.int64)
                    counts = torch.bincount(local_idx, minlength=int(item["palette"].numel()))
                    group_stats.append(
                        {
                            "palette_size": int(item["palette"].numel()),
                            "top8_cov": _coverage(counts, 8),
                            "top16_cov": _coverage(counts, 16),
                            "top32_cov": _coverage(counts, 32),
                            "top64_cov": _coverage(counts, 64),
                            "top128_cov": _coverage(counts, 128),
                            "k50": _min_k_for_threshold(counts, 0.50),
                            "k75": _min_k_for_threshold(counts, 0.75),
                            "k90": _min_k_for_threshold(counts, 0.90),
                        }
                    )
                summary = {
                    "tensor_name": tensor_name,
                    "group_rows": int(capped_rows),
                    "group_cols": int(capped_cols),
                    "launches": len(plan),
                    "mean_palette_size": sum(int(x["palette_size"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_top8_cov": sum(float(x["top8_cov"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_top16_cov": sum(float(x["top16_cov"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_top32_cov": sum(float(x["top32_cov"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_top64_cov": sum(float(x["top64_cov"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_top128_cov": sum(float(x["top128_cov"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_k50": sum(int(x["k50"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_k75": sum(int(x["k75"]) for x in group_stats) / max(len(group_stats), 1),
                    "mean_k90": sum(int(x["k90"]) for x in group_stats) / max(len(group_stats), 1),
                }
                results.append(summary)
                print(
                    f"[m6o] {tensor_name} gr={capped_rows:>4} gc={capped_cols:>4} launches={len(plan):>3} "
                    f"top32={summary['mean_top32_cov']:.3f} top64={summary['mean_top64_cov']:.3f} k90={summary['mean_k90']:.1f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"[m6o] wrote {out_path}")


if __name__ == "__main__":
    main()