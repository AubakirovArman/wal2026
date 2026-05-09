"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.selective_policy import build_selective_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-sweep-json", default=str(ROOT / "dwl2_dynamic_route/results/m6d_route_distill_depth_sweep.json"))
    parser.add_argument("--max-output-mse", type=float, default=1e-3)
    parser.add_argument("--min-local-vs-global", type=float, default=1.2)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6f_selective_runtime_policy.json"))
    args = parser.parse_args()

    with open(args.depth_sweep_json) as handle:
        depth = json.load(handle)
    policy = build_selective_policy(
        depth["records"],
        max_output_mse=args.max_output_mse,
        min_local_vs_global=args.min_local_vs_global,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2))
    print(json.dumps(policy["summary"], indent=2))
    print(f"approved tensors: {[item['tensor_name'] for item in policy['approved']]}")
    print(f"[m6f] wrote {out_path}")


if __name__ == "__main__":
    main()