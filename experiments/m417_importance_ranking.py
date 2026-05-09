"""
M417 — Importance Ranking

Ranks facts by frequency, confidence, and dependency count.
"""
import json

facts = [
    {"id": "f1", "freq": 100, "confidence": 0.99, "deps": 5},
    {"id": "f2", "freq": 50, "confidence": 0.85, "deps": 1},
    {"id": "f3", "freq": 200, "confidence": 0.90, "deps": 3},
    {"id": "f4", "freq": 10, "confidence": 0.95, "deps": 0},
    {"id": "f5", "freq": 80, "confidence": 0.80, "deps": 2},
]

# Importance score: weighted combination
for f in facts:
    f["importance"] = round(
        0.4 * (f["freq"] / 200) +
        0.3 * f["confidence"] +
        0.3 * (f["deps"] / 5),
        3
    )

facts_sorted = sorted(facts, key=lambda x: x["importance"], reverse=True)

print("=" * 60)
print("M417 — IMPORTANCE RANKING")
print("=" * 60)
print(f"{'Rank':>4} {'ID':>4} {'Freq':>6} {'Conf':>6} {'Deps':>5} {'Score':>7}")
for i, f in enumerate(facts_sorted, 1):
    print(f"{i:>4} {f['id']:>4} {f['freq']:>6} {f['confidence']:>6.2f} {f['deps']:>5} {f['importance']:>7.3f}")

assert facts_sorted[0]["id"] == "f3"  # Highest frequency
with open("experiments/m417_importance_results.json", "w") as f:
    json.dump({"ranked": facts_sorted, "top": facts_sorted[0]["id"], "pass": True}, f, indent=2)

print("\n✅ M417: Importance ranking working")
