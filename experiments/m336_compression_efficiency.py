"""
M336 — Compression Efficiency Analysis

Analyze compression ratios across different recipe types.
"""
import json

print("=" * 60)
print("M336 — COMPRESSION EFFICIENCY")
print("=" * 60)

# Different recipe types
types = {
    "short_facts": [
        {"q": "X?", "a": "Y"} for _ in range(10)
    ],
    "medium_facts": [
        {"q": f"What is the capital of country {i}?", "a": f"City {i}"} for i in range(10)
    ],
    "long_facts": [
        {"q": f"What is the detailed history and cultural significance of the capital city of country {i} including its founding date?", "a": f"A very long answer about city {i} with many details"} for i in range(10)
    ],
}

print("\nCompression analysis:")
print(f"{'Type':>15s} {'Raw':>8s} {'Delta':>8s} {'Ratio':>8s}")
print("-" * 42)

for name, recipes in types.items():
    raw = json.dumps(recipes)
    # Delta: only store first + changes
    delta = json.dumps({"base": recipes[0], "changes": recipes[1:]})
    ratio = len(raw) / len(delta)
    print(f"{name:>15s} {len(raw):>8d} {len(delta):>8d} {ratio:>7.1f}×")

results = {
    "types_analyzed": len(types),
    "best_ratio": max(len(json.dumps(r)) / len(json.dumps({"base": r[0], "changes": r[1:]})) for r in types.values()),
}

with open("experiments/m336_compression_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M336: Compression efficiency analyzed")
