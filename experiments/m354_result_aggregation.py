"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M354 — Result Aggregation

Combine all experiment results into one file.
"""
import json, os, glob

print("=" * 60)
print("M354 — RESULT AGGREGATION")
print("=" * 60)

results = {}
for rf in glob.glob("experiments/*_results.json"):
    name = os.path.basename(rf).replace("_results.json", "")
    with open(rf) as f:
        try:
            data = json.load(f)
            results[name] = data
        except:
            pass

# Aggregate
output = {
    "generated": "2026-05-03",
    "experiments": len(results),
    "results": results,
}

with open("experiments/ALL_RESULTS.json", "w") as f:
    json.dump(output, f, indent=2)

size = os.path.getsize("experiments/ALL_RESULTS.json")
print(f"\nAggregated {len(results)} results")
print(f"Output: experiments/ALL_RESULTS.json ({size} bytes)")

with open("experiments/m354_aggregate_results.json", "w") as f:
    json.dump({"experiments": len(results), "size_bytes": size}, f, indent=2)

print("\n✅ M354: Result aggregation complete")
