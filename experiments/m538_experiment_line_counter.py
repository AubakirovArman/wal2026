"""
M538 — Experiment Line Counter

Counts lines of code in experiments.
"""
import json, glob

total = 0
for path in glob.glob("experiments/m*.py"):
    with open(path) as f:
        total += len(f.readlines())

print("=" * 60)
print("M538 — LINE COUNTER")
print("=" * 60)
print(f"  Total lines: {total}")

with open("experiments/m538_lines_results.json", "w") as f:
    json.dump({"lines": total, "pass": True}, f, indent=2)

print("\n✅ M538: Line counting complete")
