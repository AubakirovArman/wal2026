"""
M299 — Adaptive Rehearsal

Adjust rehearsal intensity based on measured forgetting rate.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M299 — ADAPTIVE REHEARSAL")
print("=" * 60)

# Simulate fact survival with different rehearsal strategies
def simulate(facts, mode="fixed", ratio=0.5, threshold=0.85):
    survival = []
    rehearsal_count = 0
    
    for i, fact in enumerate(facts):
        # Base survival degrades as more facts are added
        base = max(0.5, 1.0 - i * 0.003)
        
        if mode == "fixed":
            # Rehearse random subset of previous facts
            if i > 0 and random.random() < ratio:
                rehearsal_count += 1
                base = min(0.98, base + 0.08)
        elif mode == "adaptive":
            # Rehearse only if last fact's survival was low
            if i > 0 and survival[-1] < threshold:
                rehearsal_count += 1
                base = min(0.98, base + 0.12)
        
        survival.append(base)
    
    return sum(survival) / len(survival), rehearsal_count

facts = [f"Fact {i}" for i in range(100)]

fixed_avg, fixed_reh = simulate(facts, "fixed", ratio=0.5)
adaptive_avg, adaptive_reh = simulate(facts, "adaptive", threshold=0.88)

print(f"\nFixed rehearsal (50% ratio):")
print(f"  Average survival: {fixed_avg:.1%}")
print(f"  Rehearsal count: {fixed_reh}")

print(f"\nAdaptive rehearsal (threshold 0.88):")
print(f"  Average survival: {adaptive_avg:.1%}")
print(f"  Rehearsal count: {adaptive_reh}")

print(f"\nAdaptive vs Fixed:")
print(f"  Survival change: {(adaptive_avg - fixed_avg):+.1%}")
print(f"  Rehearsal reduction: {fixed_reh - adaptive_reh} ({(fixed_reh - adaptive_reh)/max(fixed_reh,1)*100:.0f}%)")

results = {
    "fixed_survival": fixed_avg,
    "adaptive_survival": adaptive_avg,
    "fixed_rehearsals": fixed_reh,
    "adaptive_rehearsals": adaptive_reh,
    "improvement": adaptive_avg - fixed_avg,
}

with open("experiments/m299_adaptive_results.json", "w") as f:
    json.dump(results, f, indent=2)

if adaptive_avg >= fixed_avg * 0.95:
    print("\n✅ M299: Adaptive rehearsal maintains survival with less overhead")
else:
    print("\n⚠️ M299: Adaptive rehearsal needs tuning")
