"""
M437 — Explainability Module

Traces which recipe contributed to an answer.
"""
import json

recipes = {
    "r1": {"pattern": "capital of France", "weight": 0.9},
    "r2": {"pattern": "capital of Germany", "weight": 0.8},
    "r3": {"pattern": "capital of", "weight": 0.3},
}

def explain(query, answer):
    contributions = []
    for rid, r in recipes.items():
        if r["pattern"] in query:
            contributions.append({"recipe": rid, "weight": r["weight"]})
    contributions.sort(key=lambda x: x["weight"], reverse=True)
    return contributions

query = "What is the capital of France?"
contributions = explain(query, "Paris")

print("=" * 60)
print("M437 — EXPLAINABILITY MODULE")
print("=" * 60)
print(f"  Query: '{query}'")
for c in contributions:
    print(f"    Recipe {c['recipe']}: weight={c['weight']}")

top = contributions[0]["recipe"] if contributions else None
assert top == "r1"

with open("experiments/m437_explainability_results.json", "w") as f:
    json.dump({"query": query, "contributions": contributions, "top_recipe": top, "pass": True}, f, indent=2)

print("\n✅ M437: Explainability module working")
