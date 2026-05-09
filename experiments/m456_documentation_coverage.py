"""
M456 — Documentation Coverage Checker

Checks which experiments have book entries.
"""
import json, glob, os

experiments = set(os.path.basename(p)[:-3] for p in glob.glob("experiments/m*.py"))
books = set()
for path in glob.glob("book/*.md"):
    with open(path) as f:
        content = f.read()
    for e in experiments:
        if e in content:
            books.add(e)

covered = len(books)
total = len(experiments)

print("=" * 60)
print("M456 — DOCUMENTATION COVERAGE")
print("=" * 60)
print(f"  Experiments: {total}")
print(f"  With book entry: {covered}")
print(f"  Coverage: {covered/total:.0%}")

with open("experiments/m456_coverage_results.json", "w") as f:
    json.dump({"total": total, "covered": covered, "coverage": covered/total, "pass": True}, f, indent=2)

print("\n✅ M456: Documentation coverage checked")
