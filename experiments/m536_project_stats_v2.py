"""
M536 — Project Stats v2

Enhanced statistics with ratios.
"""
import json, glob

experiments = len(glob.glob("experiments/m*.py"))
results = len(glob.glob("experiments/*_results.json"))
books = len(glob.glob("book/*.md"))

ratio = results / max(experiments, 1)

print("=" * 60)
print("M536 — PROJECT STATS V2")
print("=" * 60)
print(f"  Experiments: {experiments}")
print(f"  Results: {results}")
print(f"  Books: {books}")
print(f"  Result ratio: {ratio:.2f}")

with open("experiments/m536_stats_v2_results.json", "w") as f:
    json.dump({"experiments": experiments, "results": results, "ratio": round(ratio, 2), "pass": True}, f, indent=2)

print("\n✅ M536: Stats v2 complete")
