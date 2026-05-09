"""
M537 — Result Size Analyzer

Analyzes size of result files.
"""
import json, glob, os

sizes = []
for path in glob.glob("experiments/*_results.json"):
    sizes.append(os.path.getsize(path))

avg = sum(sizes) / max(len(sizes), 1)

print("=" * 60)
print("M537 — RESULT SIZE ANALYZER")
print("=" * 60)
print(f"  Files: {len(sizes)}")
print(f"  Avg size: {avg:.0f} bytes")
print(f"  Total: {sum(sizes) / 1024:.1f} KB")

with open("experiments/m537_size_results.json", "w") as f:
    json.dump({"files": len(sizes), "avg_bytes": round(avg, 1), "pass": True}, f, indent=2)

print("\n✅ M537: Result size analysis complete")
