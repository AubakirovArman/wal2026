"""M622 — Result Schema Gate.

Runs the WAL result validator over all experiment result JSON files.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wal.results import validate_results

summary = validate_results(ROOT / "experiments")
payload = summary.to_dict()
payload.update({
    "experiment": "M622",
    "gate": "result_schema",
})

print("=" * 60)
print("M622 — RESULT SCHEMA GATE")
print("=" * 60)
print(f"  Valid: {payload['valid']}/{payload['total']}")
print(f"  Invalid: {payload['invalid']}")
print(f"  Warnings: {payload['warnings']}")

(ROOT / "experiments/m622_result_schema_gate_results.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)
print(f"\nM622 status={payload['status']}")
