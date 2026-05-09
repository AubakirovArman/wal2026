"""
M558 — Experiment Count Badge

Generates experiment count badge.
"""
import json, glob

count = len(glob.glob("experiments/m*.py"))
badge = f"![Experiments](https://img.shields.io/badge/experiments-{count}-blue)"

print("=" * 60)
print("M558 — EXP COUNT BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m558_exp_badge_results.json", "w") as f:
    json.dump({"count": count, "pass": True}, f, indent=2)

print("\n✅ M558: Experiment count badge generated")
