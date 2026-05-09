"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M460 — Project Health Score

Computes composite health score from multiple dimensions.
"""
import json, glob, os

# Collect metrics
experiments = len(glob.glob("experiments/m*.py"))
results = len(glob.glob("experiments/*_results.json"))
books = len(glob.glob("book/*.md"))

# Pass rate
passing = 0
total_checks = 0
for path in glob.glob("experiments/m4*_results.json"):
    with open(path) as f:
        data = json.load(f)
    total_checks += 1
    if data.get("pass"):
        passing += 1

pass_rate = passing / max(total_checks, 1)

# Health score: weighted composite
score = (
    0.3 * min(experiments / 500, 1.0) +
    0.3 * pass_rate +
    0.2 * min(results / 200, 1.0) +
    0.2 * min(books / 300, 1.0)
)

print("=" * 60)
print("M460 — PROJECT HEALTH SCORE")
print("=" * 60)
print(f"  Experiments: {experiments}")
print(f"  Pass rate: {pass_rate:.0%}")
print(f"  Results: {results}")
print(f"  Books: {books}")
print(f"\n  Health Score: {score:.2f}/1.00")

grade = "A+" if score > 0.95 else "A" if score > 0.85 else "B"
print(f"  Grade: {grade}")

with open("experiments/m460_health_score_results.json", "w") as f:
    json.dump({"score": round(score, 3), "grade": grade, "pass": score > 0.8}, f, indent=2)

print("\n✅ M460: Project health score calculated")
