"""
M510 — Naming Convention Check

Validates experiment naming consistency.
"""
import json, glob, re, os

valid = 0
invalid = 0
for path in glob.glob("experiments/m*.py"):
    name = re.match(r"m\d{3}_[a-z_0-9]+\.py", os.path.basename(path))
    if name:
        valid += 1
    else:
        invalid += 1

print("=" * 60)
print("M510 — NAMING CONVENTION")
print("=" * 60)
print(f"  Valid: {valid}")
print(f"  Invalid: {invalid}")

with open("experiments/m510_naming_results.json", "w") as f:
    json.dump({"valid": valid, "invalid": invalid, "pass": invalid == 0}, f, indent=2)

print("\n✅ M510: Naming convention check complete")
