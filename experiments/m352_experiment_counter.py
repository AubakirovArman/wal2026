"""
M352 — Experiment Counter

Categorize and count all experiments.
"""
import json, os

print("=" * 60)
print("M352 — EXPERIMENT COUNTER")
print("=" * 60)

files = [f for f in os.listdir("experiments") if f.endswith(".py")]

categories = {
    "core": ["m251", "m252", "m253", "m254", "m255", "m256"],
    "ci": ["m265", "m270", "m281", "m282", "m283", "m284"],
    "scale": ["m276", "m277", "m278", "m279", "m280", "m295", "m300"],
    "deployment": ["m301", "m304", "m308", "m309", "m310"],
    "safety": ["m311", "m312", "m313", "m314", "m315"],
    "advanced": ["m316", "m317", "m318", "m319", "m320"],
    "optimization": ["m291", "m306", "m342", "m338"],
    "wild": ["recipe", "evolution", "immune", "canary", "fuzzing"],
}

counts = {}
for cat, prefixes in categories.items():
    count = sum(1 for f in files if any(p in f.lower() for p in prefixes))
    counts[cat] = count

print("\nExperiment categories:")
for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (count // 2)
    print(f"  {cat:>15s}: {count:>3d} {bar}")

print(f"\n  {'Total':>15s}: {len(files):>3d}")

with open("experiments/m352_counter_results.json", "w") as f:
    json.dump({"total": len(files), "categories": counts}, f, indent=2)

print("\n✅ M352: Experiment counter complete")
