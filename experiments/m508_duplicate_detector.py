"""
M508 — Duplicate Detector

Finds duplicate experiment names.
"""
import json, glob, os
from collections import Counter

names = [os.path.basename(p) for p in glob.glob("experiments/m*.py")]
dups = {k: v for k, v in Counter(names).items() if v > 1}

print("=" * 60)
print("M508 — DUPLICATE DETECTOR")
print("=" * 60)
print(f"  Duplicates: {len(dups)}")

with open("experiments/m508_duplicate_results.json", "w") as f:
    json.dump({"duplicates": len(dups), "pass": len(dups) == 0}, f, indent=2)

print("\n✅ M508: Duplicate detection complete")
