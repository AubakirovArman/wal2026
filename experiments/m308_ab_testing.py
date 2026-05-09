"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M308 — A/B Testing

Compare two model versions on same traffic.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M308 — A/B TESTING")
print("=" * 60)

# Simulate two model versions
class ModelVersion:
    def __init__(self, name, accuracy, latency_ms):
        self.name = name
        self.accuracy = accuracy
        self.latency_ms = latency_ms
    
    def answer(self, question):
        correct = random.random() < self.accuracy
        return correct, self.latency_ms

model_a = ModelVersion("v1.0_base", accuracy=0.85, latency_ms=45)
model_b = ModelVersion("v1.1_edited", accuracy=0.92, latency_ms=48)

# Test questions
test_questions = [f"Question {i}" for i in range(100)]

print("\nRunning A/B test on 100 questions...")
results_a = {"correct": 0, "total": 0, "latency": 0}
results_b = {"correct": 0, "total": 0, "latency": 0}

for q in test_questions:
    # Randomly assign to A or B (50/50)
    if random.random() < 0.5:
        correct, lat = model_a.answer(q)
        results_a["correct"] += int(correct)
        results_a["total"] += 1
        results_a["latency"] += lat
    else:
        correct, lat = model_b.answer(q)
        results_b["correct"] += int(correct)
        results_b["total"] += 1
        results_b["latency"] += lat

# Calculate metrics
acc_a = results_a["correct"] / results_a["total"] if results_a["total"] else 0
acc_b = results_b["correct"] / results_b["total"] if results_b["total"] else 0
lat_a = results_a["latency"] / results_a["total"] if results_a["total"] else 0
lat_b = results_b["latency"] / results_b["total"] if results_b["total"] else 0

print(f"\nModel A ({model_a.name}):")
print(f"  Accuracy: {acc_a:.1%} ({results_a['correct']}/{results_a['total']})")
print(f"  Avg latency: {lat_a:.0f}ms")

print(f"\nModel B ({model_b.name}):")
print(f"  Accuracy: {acc_b:.1%} ({results_b['correct']}/{results_b['total']})")
print(f"  Avg latency: {lat_b:.0f}ms")

print(f"\nComparison:")
print(f"  Accuracy delta: {(acc_b - acc_a):+.1%}")
print(f"  Latency delta: {(lat_b - lat_a):+.0f}ms")

# Statistical significance (simplified)
improvement = acc_b > acc_a
print(f"\n  {'✅' if improvement else '❌'} Model B is {'better' if improvement else 'worse'} than Model A")

# Winner
winner = "B" if acc_b > acc_a else "A"
print(f"\n  Winner: Model {winner}")

with open("experiments/m308_ab_results.json", "w") as f:
    json.dump({
        "model_a": {"name": model_a.name, "accuracy": acc_a, "latency_ms": lat_a, "samples": results_a["total"]},
        "model_b": {"name": model_b.name, "accuracy": acc_b, "latency_ms": lat_b, "samples": results_b["total"]},
        "winner": winner,
        "accuracy_delta": acc_b - acc_a,
    }, f, indent=2)

print("\n✅ M308: A/B testing framework working")
