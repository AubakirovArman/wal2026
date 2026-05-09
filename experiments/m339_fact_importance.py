"""
M339 — Fact Importance Ranking

Rank facts by usage frequency and criticality.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M339 — FACT IMPORTANCE RANKING")
print("=" * 60)

# Facts with usage stats
facts = [
    {"id": 1, "question": "Capital of France?", "uses": 150, "critical": True},
    {"id": 2, "question": "Capital of Japan?", "uses": 120, "critical": True},
    {"id": 3, "question": "Capital of obscure country?", "uses": 5, "critical": False},
    {"id": 4, "question": "Speed of light?", "uses": 200, "critical": True},
    {"id": 5, "question": "H2O formula?", "uses": 80, "critical": False},
]

# Calculate importance score
for f in facts:
    f["importance"] = f["uses"] * (2 if f["critical"] else 1)

# Sort by importance
facts.sort(key=lambda x: x["importance"], reverse=True)

print("\nFact importance ranking:")
print(f"{'Rank':>5s} {'ID':>4s} {'Question':>30s} {'Uses':>6s} {'Critical':>10s} {'Score':>8s}")
print("-" * 70)
for i, f in enumerate(facts):
    crit = "YES" if f["critical"] else "no"
    print(f"{i+1:>5d} {f['id']:>4d} {f['question']:>30s} {f['uses']:>6d} {crit:>10s} {f['importance']:>8d}")

# Top facts to protect
top_facts = facts[:3]
print(f"\nTop {len(top_facts)} facts to protect:")
for f in top_facts:
    print(f"  [{f['id']}] {f['question']} (score: {f['importance']})")

with open("experiments/m339_importance_results.json", "w") as f:
    json.dump({
        "facts_ranked": len(facts),
        "top_facts": len(top_facts),
        "highest_score": facts[0]["importance"] if facts else 0,
    }, f, indent=2)

print("\n✅ M339: Fact importance ranking complete")
