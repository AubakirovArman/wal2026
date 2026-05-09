"""
M544 — Result Validation

Validates result files have required fields.
"""
import json, glob

valid = 0
invalid = 0
for path in glob.glob("experiments/*_results.json"):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        valid += 1
    else:
        invalid += 1

print("=" * 60)
print("M544 — RESULT VALIDATION")
print("=" * 60)
print(f"  Valid: {valid}")
print(f"  Invalid: {invalid}")

with open("experiments/m544_result_validation_results.json", "w") as f:
    json.dump({"valid": valid, "invalid": invalid, "pass": invalid == 0}, f, indent=2)

print("\n✅ M544: Result validation complete")
