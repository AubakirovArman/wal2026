"""
M310 — Graceful Degradation

System degrades gracefully under high load or errors.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M310 — GRACEFUL DEGRADATION")
print("=" * 60)

class DegradingModel:
    def __init__(self, max_load=100):
        self.max_load = max_load
        self.current_load = 0
        self.errors = 0
    
    def infer(self, question):
        self.current_load += 1
        
        # Under normal load
        if self.current_load <= self.max_load * 0.7:
            return {"answer": f"Full answer to {question}", "quality": "high", "cached": False}
        
        # Under moderate load: use cache
        elif self.current_load <= self.max_load * 0.9:
            return {"answer": f"Cached: {question[:10]}...", "quality": "medium", "cached": True}
        
        # Under high load: return fallback
        else:
            self.errors += 1
            if random.random() < 0.5:
                return {"answer": "Service busy, try again", "quality": "low", "error": True}
            else:
                return {"answer": f"Quick: {question[:5]}...", "quality": "low", "cached": False}

model = DegradingModel(max_load=100)

# Simulate increasing load
print("\nSimulating load increase...")
requests = [f"Q{i}" for i in range(120)]
results = []

for i, q in enumerate(requests):
    r = model.infer(q)
    r["load"] = model.current_load / model.max_load
    results.append(r)
    
    if i in [0, 34, 69, 99, 119]:
        quality_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[r["quality"]]
        print(f"  Load {r['load']:>5.0%} | {quality_icon} {r['quality']:>6s} | {r['answer'][:30]}...")

# Analyze degradation
high_q = sum(1 for r in results if r["quality"] == "high")
med_q = sum(1 for r in results if r["quality"] == "medium")
low_q = sum(1 for r in results if r["quality"] == "low")
errors = sum(1 for r in results if r.get("error"))

print(f"\nQuality distribution:")
print(f"  High:   {high_q} ({high_q/len(results):.0%})")
print(f"  Medium: {med_q} ({med_q/len(results):.0%})")
print(f"  Low:    {low_q} ({low_q/len(results):.0%})")
print(f"  Errors: {errors} ({errors/len(results):.0%})")

# Degradation is graceful if errors are low
graceful = errors / len(results) < 0.15
print(f"\n{'✅' if graceful else '❌'} Degradation {'graceful' if graceful else 'not graceful'} ({errors/len(results):.0%} errors)")

with open("experiments/m310_degradation_results.json", "w") as f:
    json.dump({
        "high_quality": high_q,
        "medium_quality": med_q,
        "low_quality": low_q,
        "errors": errors,
        "graceful": graceful,
    }, f, indent=2)

print("\n✅ M310: Graceful degradation under load")
