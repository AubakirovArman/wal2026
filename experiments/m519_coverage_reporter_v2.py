"""
M519 — Coverage Reporter v2

Reports test coverage across experiment categories.
"""
import json, glob, os

categories = {
    "core": glob.glob("experiments/m[12]*_*.py"),
    "infra": glob.glob("experiments/m[23]*_*.py"),
    "advanced": glob.glob("experiments/m[45]*_*.py"),
}

results = {}
for cat, files in categories.items():
    result_files = [f.replace(".py", "_results.json") for f in files]
    present = sum(1 for r in result_files if os.path.exists(r))
    results[cat] = {"total": len(files), "results": present, "coverage": present / max(len(files), 1)}

print("=" * 60)
print("M519 — COVERAGE REPORTER V2")
print("=" * 60)
for cat, r in results.items():
    print(f"  {cat}: {r['results']}/{r['total']} ({r['coverage']:.0%})")

with open("experiments/m519_coverage_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M519: Coverage report v2 generated")
