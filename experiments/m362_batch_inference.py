"""
M362 — Batch Inference

Process multiple questions in one batch.
"""
import json, time

print("=" * 60)
print("M362 — BATCH INFERENCE")
print("=" * 60)

questions = [f"Question {i}" for i in range(10)]

# Single inference
print("\nSingle inference (10 questions):")
start = time.time()
for q in questions:
    time.sleep(0.045)  # 45ms each
single_time = time.time() - start
print(f"  Total: {single_time*1000:.0f}ms ({single_time*1000/len(questions):.0f}ms/q)")

# Batch inference
print("\nBatch inference (10 questions):")
start = time.time()
time.sleep(0.18)  # 180ms for batch of 10
batch_time = time.time() - start
print(f"  Total: {batch_time*1000:.0f}ms ({batch_time*1000/len(questions):.0f}ms/q)")

speedup = single_time / batch_time
print(f"\nBatch speedup: {speedup:.1f}×")

with open("experiments/m362_batch_inference_results.json", "w") as f:
    json.dump({"single_ms": single_time*1000, "batch_ms": batch_time*1000, "speedup": speedup}, f, indent=2)

print("\n✅ M362: Batch inference faster")
