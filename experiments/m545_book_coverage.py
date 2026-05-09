"""
M545 — Book Coverage by Phase

Checks which phases have book entries.
"""
import json, glob, re, os

phases = {"m1": 0, "m2": 0, "m3": 0, "m4": 0, "m5": 0}
for path in glob.glob("book/*.md"):
    with open(path) as f:
        content = f.read()
    for phase in phases:
        if re.search(rf"{phase.upper()}", content):
            phases[phase] += 1

print("=" * 60)
print("M545 — BOOK COVERAGE")
print("=" * 60)
for phase, count in phases.items():
    print(f"  {phase}: {count} mentions")

with open("experiments/m545_book_coverage_results.json", "w") as f:
    json.dump(phases, f, indent=2)

print("\n✅ M545: Book coverage analyzed")
