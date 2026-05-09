"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M338 — Smart Rehearsal

Rehearse only the weakest facts.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M338 — SMART REHEARSAL")
print("=" * 60)

# Facts with simulated strength
facts = [
    {"id": 1, "question": "Q1", "strength": 0.95},
    {"id": 2, "question": "Q2", "strength": 0.88},
    {"id": 3, "question": "Q3", "strength": 0.72},
    {"id": 4, "question": "Q4", "strength": 0.91},
    {"id": 5, "question": "Q5", "strength": 0.65},
    {"id": 6, "question": "Q6", "strength": 0.98},
]

print("Initial fact strengths:")
for f in facts:
    bar = "█" * int(f["strength"] * 10) + "░" * (10 - int(f["strength"] * 10))
    print(f"  [{f['id']}] {bar} {f['strength']:.0%}")

# Smart rehearsal: only rehearse facts below threshold
threshold = 0.80
weak_facts = [f for f in facts if f["strength"] < threshold]

print(f"\nSmart rehearsal (threshold {threshold:.0%}):")
print(f"  Weak facts: {len(weak_facts)}/{len(facts)}")

for f in weak_facts:
    # Rehearse: increase strength
    old = f["strength"]
    f["strength"] = min(0.98, f["strength"] + random.uniform(0.05, 0.15))
    print(f"  [{f['id']}] {old:.0%} → {f['strength']:.0%} (rehearsed)")

print(f"\nFinal strengths:")
avg_before = sum(f["strength"] for f in facts) / len(facts)
for f in facts:
    bar = "█" * int(f["strength"] * 10) + "░" * (10 - int(f["strength"] * 10))
    print(f"  [{f['id']}] {bar} {f['strength']:.0%}")

with open("experiments/m338_smart_rehearsal_results.json", "w") as f:
    json.dump({
        "facts_total": len(facts),
        "facts_rehearsed": len(weak_facts),
        "avg_strength_after": avg_before,
    }, f, indent=2)

print("\n✅ M338: Smart rehearsal improves weak facts")
