from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.selective_policy import build_shape_runtime_policy


def _load_results(paths: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        results.extend(payload["results"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-json", default=str(ROOT / "dwl2_dynamic_route/results/m6f_selective_runtime_policy.json"))
    parser.add_argument(
        "--bench-json",
        nargs="+",
        default=[
            str(ROOT / "dwl2_dynamic_route/results/m6l_fp16_local_palette_frontier_bench.json"),
            str(ROOT / "dwl2_dynamic_route/results/m6l_fp16_local_palette_frontier_bench_rerun.json"),
            str(ROOT / "dwl2_dynamic_route/results/m6q_larger_row_groups_bench.json"),
            str(ROOT / "dwl2_dynamic_route/results/m6q_larger_row_groups_bench_rerun.json"),
            str(ROOT / "dwl2_dynamic_route/results/m6r_ultra_row_groups_bench.json"),
        ],
    )
    parser.add_argument("--max-grouped-local-mse", type=float, default=1e-4)
    parser.add_argument("--min-local-vs-global-full", type=float, default=1.0)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6s_shape_runtime_policy.json"))
    args = parser.parse_args()

    with open(args.policy_json) as handle:
        policy = json.load(handle)
    runtime_policy = build_shape_runtime_policy(
        policy,
        _load_results(args.bench_json),
        max_grouped_local_mse=args.max_grouped_local_mse,
        min_local_vs_global_full=args.min_local_vs_global_full,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(runtime_policy, indent=2))
    print(json.dumps(runtime_policy["summary"], indent=2))
    print(f"selected tensors: {[item['tensor_name'] for item in runtime_policy['selected']]}")
    print(f"[m6s] wrote {out_path}")


if __name__ == "__main__":
    main()