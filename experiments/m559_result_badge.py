"""
M559 — Result Count Badge

Generates result count badge.
"""
import json, glob

count = len(glob.glob("experiments/*_results.json"))
badge = f"![Results](https://img.shields.io/badge/results-{count}-blue)"

print("=" * 60)
print("M559 — RESULT BADGE")
print("=" * 60)
print(f"  {badge}")

with open("experiments/m559_result_badge_results.json", "w") as f:
    json.dump({"count": count, "pass": True}, f, indent=2)

print("\n✅ M559: Result badge generated")
