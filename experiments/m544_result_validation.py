"""M544 — Result Validation.

Validates result files against the WAL result schema.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wal.results import validate_results

print("=" * 60)
print("M544 — RESULT VALIDATION")
print("=" * 60)
summary = validate_results("experiments")
payload = summary.to_dict()
print(f"  Valid: {payload['valid']}")
print(f"  Invalid: {payload['invalid']}")
print(f"  Warnings: {payload['warnings']}")

with open("experiments/m544_result_validation_results.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nM544: Result validation status={payload['status']}")
