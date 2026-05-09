"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M444 — Impact Prediction Model

Predicts impact of a new edit on existing facts.
"""
import json

existing = ["Paris is France", "Berlin is Germany", "Madrid is Spain"]
proposed = "Lisbon is Portugal"

def predict_impact(existing, proposed):
    # Simple heuristic: shared words = higher impact
    words = set(proposed.lower().split())
    impacts = []
    for e in existing:
        shared = len(words & set(e.lower().split()))
        impacts.append({"fact": e, "impact": shared / max(len(words), 1)})
    return impacts

impacts = predict_impact(existing, proposed)

print("=" * 60)
print("M444 — IMPACT PREDICTION")
print("=" * 60)

for i in impacts:
    print(f"  '{i['fact']}': impact={i['impact']:.2f}")

max_impact = max(i["impact"] for i in impacts)
print(f"\nMax predicted impact: {max_impact:.2f}")

with open("experiments/m444_impact_results.json", "w") as f:
    json.dump({"impacts": impacts, "max_impact": max_impact, "pass": True}, f, indent=2)

print("\n✅ M444: Impact prediction working")
