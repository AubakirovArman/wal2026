"""
M306 — Response Caching

Cache frequent answers to reduce inference load.
"""
import json, time

print("=" * 60)
print("M306 — RESPONSE CACHING")
print("=" * 60)

class CachedModel:
    def __init__(self, cache_size=100):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = cache_size
    
    def infer(self, question):
        # Check cache
        if question in self.cache:
            self.cache_hits += 1
            return self.cache[question], True
        
        # Simulate inference
        self.cache_misses += 1
        time.sleep(0.001)  # 1ms inference
        
        # Generate answer (mock)
        answer = f"Answer to: {question[:20]}..."
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[question] = answer
        
        return answer, False

model = CachedModel(cache_size=50)

# Simulate traffic with repeated questions
questions = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of France?",  # repeat
    "What is the capital of Brazil?",
    "What is the capital of Japan?",  # repeat
    "What is the capital of France?",  # repeat
    "What is the capital of Egypt?",
    "What is the capital of Brazil?",  # repeat
]

print("\nSimulating inference with caching...")
for q in questions:
    ans, cached = model.infer(q)
    status = "CACHE" if cached else "INFER"
    print(f"  [{status}] {q[:35]}... → {ans[:25]}")

total = model.cache_hits + model.cache_misses
hit_rate = model.cache_hits / total if total else 0

print(f"\nCache statistics:")
print(f"  Hits: {model.cache_hits}")
print(f"  Misses: {model.cache_misses}")
print(f"  Hit rate: {hit_rate:.1%}")
print(f"  Cache size: {len(model.cache)}/{model.cache_size}")

# Calculate latency improvement
infer_time = 1.0  # ms
cache_time = 0.01  # ms
avg_time = (model.cache_hits * cache_time + model.cache_misses * infer_time) / total
print(f"  Avg latency: {avg_time:.2f}ms (vs {infer_time:.2f}ms without cache)")
print(f"  Speedup: {infer_time / avg_time:.1f}×")

results = {
    "hits": model.cache_hits,
    "misses": model.cache_misses,
    "hit_rate": hit_rate,
    "avg_latency_ms": avg_time,
    "speedup": infer_time / avg_time,
}

with open("experiments/m306_caching_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M306: Response caching improves latency")
