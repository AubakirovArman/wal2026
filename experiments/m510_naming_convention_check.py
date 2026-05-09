"""
M510 — Naming Convention Check

Validates experiment naming consistency.
"""
import json, glob, re, os

valid = 0
invalid = 0
legacy_named = 0
invalid_files = []
milestone_pattern = re.compile(r"m\d+[a-z]*_[a-z0-9_]+\.py")
legacy_named_pattern = re.compile(r"[a-z][a-z0-9_]+\.py")
for path in glob.glob("experiments/m*.py"):
    basename = os.path.basename(path)
    if milestone_pattern.fullmatch(basename):
        valid += 1
    elif legacy_named_pattern.fullmatch(basename):
        legacy_named += 1
    else:
        invalid += 1
        invalid_files.append(basename)

print("=" * 60)
print("M510 — NAMING CONVENTION")
print("=" * 60)
print(f"  Valid: {valid}")
print(f"  Legacy named: {legacy_named}")
print(f"  Invalid: {invalid}")

with open("experiments/m510_naming_results.json", "w") as f:
    json.dump({
        "schema_version": "wal.results.v1",
        "valid": valid,
        "legacy_named": legacy_named,
        "invalid": invalid,
        "invalid_files": invalid_files,
        "status": "PASS" if invalid == 0 else "FAIL",
        "pass": invalid == 0,
    }, f, indent=2)

print(f"\nM510: Naming convention status={'PASS' if invalid == 0 else 'FAIL'}")
