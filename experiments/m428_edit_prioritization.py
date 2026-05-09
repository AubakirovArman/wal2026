"""
M428 — Edit Prioritization

Ranks pending edits by urgency and impact.
"""
import json

edits = [
    {"id": "e1", "urgency": 5, "impact": 100, "risk": 2},
    {"id": "e2", "urgency": 3, "impact": 50, "risk": 1},
    {"id": "e3", "urgency": 5, "impact": 200, "risk": 5},
    {"id": "e4", "urgency": 1, "impact": 10, "risk": 1},
]

# Priority score: urgency * impact / risk
for e in edits:
    e["priority"] = round((e["urgency"] * e["impact"]) / max(e["risk"], 1), 2)

edits_sorted = sorted(edits, key=lambda x: x["priority"], reverse=True)

print("=" * 60)
print("M428 — EDIT PRIORITIZATION")
print("=" * 60)
print(f"{'Rank':>4} {'ID':>4} {'Urgency':>8} {'Impact':>8} {'Risk':>6} {'Priority':>10}")
for i, e in enumerate(edits_sorted, 1):
    print(f"{i:>4} {e['id']:>4} {e['urgency']:>8} {e['impact']:>8} {e['risk']:>6} {e['priority']:>10.2f}")

assert edits_sorted[0]["id"] == "e1"  # Best urgency/impact/risk ratio
with open("experiments/m428_prioritization_results.json", "w") as f:
    json.dump({"ranked": edits_sorted, "top": edits_sorted[0]["id"], "pass": True}, f, indent=2)

print("\n✅ M428: Edit prioritization working")
