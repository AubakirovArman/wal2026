"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M373 — Fact Analytics

Analytics dashboard for facts.
"""
import json

print("=" * 60)
print("M373 — FACT ANALYTICS")
print("=" * 60)

facts = [
    {"domain": "geo", "correct": 150, "wrong": 10},
    {"domain": "sci", "correct": 80, "wrong": 5},
    {"domain": "hist", "correct": 60, "wrong": 15},
]

print("\nFact analytics:")
print(f"{'Domain':>8s} {'Correct':>8s} {'Wrong':>8s} {'Accuracy':>10s}")
print("-" * 38)

total_correct = 0
total_wrong = 0
for f in facts:
    acc = f["correct"] / (f["correct"] + f["wrong"])
    total_correct += f["correct"]
    total_wrong += f["wrong"]
    print(f"{f['domain']:>8s} {f['correct']:>8d} {f['wrong']:>8d} {acc:>9.1%}")

total_acc = total_correct / (total_correct + total_wrong)
print(f"\n{'Total':>8s} {total_correct:>8d} {total_wrong:>8d} {total_acc:>9.1%}")

with open("experiments/m373_analytics_results.json", "w") as f:
    json.dump({"domains": len(facts), "total_accuracy": total_acc}, f, indent=2)

print("\n✅ M373: Fact analytics complete")
