from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "corpora" / "ci_score_calibration.json"
RESULT_PATH = ROOT / "experiments" / "m650_ci_score_calibration_results.json"


WEIGHTS = {
    "exact": 0.35,
    "negative": 0.25,
    "context": 0.20,
    "ppl": 0.10,
    "refusal": 0.10,
}

PASS_THRESHOLD = 0.85
WARN_THRESHOLD = 0.70
CRITICAL_FLOORS = {
    "exact": 0.80,
    "negative": 0.60,
    "context": 0.40,
    "refusal": 0.70,
}


def score(metrics: dict[str, float]) -> float:
    return round(sum(metrics[key] * weight for key, weight in WEIGHTS.items()), 4)


def verdict(value: float, metrics: dict[str, float]) -> str:
    if any(metrics[key] < floor for key, floor in CRITICAL_FLOORS.items()):
        return "FAIL"
    if value >= PASS_THRESHOLD:
        return "PASS"
    if value >= WARN_THRESHOLD:
        return "WARN"
    return "FAIL"


def main() -> int:
    scenarios = [
        {
            "name": "healthy_release",
            "metrics": {"exact": 1.0, "negative": 1.0, "context": 0.95, "ppl": 0.95, "refusal": 1.0},
            "expected_verdict": "PASS",
        },
        {
            "name": "negative_regression",
            "metrics": {"exact": 0.95, "negative": 0.30, "context": 0.90, "ppl": 0.95, "refusal": 1.0},
            "expected_verdict": "FAIL",
        },
        {
            "name": "context_warning",
            "metrics": {"exact": 0.90, "negative": 0.90, "context": 0.45, "ppl": 0.95, "refusal": 0.90},
            "expected_verdict": "WARN",
        },
    ]
    calibrated = []
    failures = []
    for scenario in scenarios:
        value = score(scenario["metrics"])
        observed = verdict(value, scenario["metrics"])
        calibrated.append({**scenario, "score": value, "observed_verdict": observed})
        if observed != scenario["expected_verdict"]:
            failures.append({"name": scenario["name"], "expected": scenario["expected_verdict"], "observed": observed})

    weight_sum = round(sum(WEIGHTS.values()), 6)
    if weight_sum != 1.0:
        failures.append({"name": "weights", "expected": "1.0", "observed": str(weight_sum)})

    payload = {
        "weights": WEIGHTS,
        "pass_threshold": PASS_THRESHOLD,
        "warn_threshold": WARN_THRESHOLD,
        "critical_floors": CRITICAL_FLOORS,
        "scenarios": calibrated,
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M650",
        "name": "CI Score Calibration",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": WEIGHTS,
        "pass_threshold": PASS_THRESHOLD,
        "warn_threshold": WARN_THRESHOLD,
        "critical_floors": CRITICAL_FLOORS,
        "scenarios": calibrated,
        "failures": failures,
        "calibration": str(CALIBRATION_PATH.relative_to(ROOT)),
        "scope": "CI scoring policy calibration only",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M650 CI Score Calibration: {status}")
    print(f"scenarios={len(scenarios)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
