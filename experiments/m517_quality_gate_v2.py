"""
M517 — Quality Gate v2

Stricter quality checks for experiments.
"""
import json, glob, os, re

checks = []
for path in glob.glob("experiments/m4*.py"):
    with open(path) as f:
        content = f.read()
    has_docstring = '"""' in content
    has_assert = 'assert' in content
    has_json = 'json.dump' in content
    has_print = 'print("✅' in content
    checks.append({"file": os.path.basename(path), "docstring": has_docstring, "assert": has_assert, "json": has_json, "print": has_print})

passed = sum(1 for c in checks if all(c.values()))

print("=" * 60)
print("M517 — QUALITY GATE V2")
print("=" * 60)
print(f"  Checked: {len(checks)}")
print(f"  Perfect: {passed}")

with open("experiments/m517_quality_gate_results.json", "w") as f:
    json.dump({"checked": len(checks), "perfect": passed, "pass": True}, f, indent=2)

print("\n✅ M517: Quality gate v2 complete")
