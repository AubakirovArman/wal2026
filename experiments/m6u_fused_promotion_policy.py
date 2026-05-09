from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _suffix_counts(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        suffix = name.split(".", 3)[-1] if name.count(".") >= 3 else name
        counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaptive-gate-json",
        default=str(ROOT / "dwl2_dynamic_route/results/m6t_selective_runtime_gate_adaptive_shadow4calls_4w_fixed.json"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "dwl2_dynamic_route/results/m6u_fused_promotion_policy.json"),
    )
    args = parser.parse_args()

    with open(args.adaptive_gate_json) as handle:
        gate = json.load(handle)
    adaptive_names = gate.get("adaptive_names", {})
    promoted_layers = sorted(str(name) for name in adaptive_names.get("primary_enabled_layer_names", []))
    fallback_layers = sorted(str(name) for name in adaptive_names.get("fallback_layer_names", []))
    shadow_mismatch_layers = sorted(str(name) for name in adaptive_names.get("shadow_mismatch_layer_names", []))
    policy = {
        "source_gate": args.adaptive_gate_json,
        "promoted_layers": promoted_layers,
        "rejected": {
            "fallback_layers": fallback_layers,
            "shadow_mismatch_layers": shadow_mismatch_layers,
        },
        "summary": {
            "promoted_count": len(promoted_layers),
            "fallback_count": len(fallback_layers),
            "shadow_mismatch_count": len(shadow_mismatch_layers),
            "promoted_suffix_counts": _suffix_counts(promoted_layers),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2))
    print(json.dumps(policy["summary"], indent=2))
    print(f"[m6u] wrote {out_path}")


if __name__ == "__main__":
    main()