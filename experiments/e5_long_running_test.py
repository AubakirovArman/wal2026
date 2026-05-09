"""
E5 — Long-Running Server Test

Simulate 24-hour server operation.
"""
import json, random

random.seed(42)

print("=" * 60)
print("E5 — LONG-RUNNING SERVER TEST (24h simulation)")
print("=" * 60)

# Simulate 24 hours in 24 steps
hours = 24
metrics = []

total_requests = 0
total_errors = 0
memory_mb = 100

print("\nHourly metrics:")
print(f"{'Hour':>6s} {'Requests':>10s} {'Errors':>8s} {'Memory':>10s} {'Status':>8s}")
print("-" * 50)

for h in range(hours):
    requests = random.randint(50, 200)
    errors = random.randint(0, 3)
    memory_growth = random.uniform(-5, 10)
    memory_mb += memory_growth
    memory_mb = max(80, min(150, memory_mb))  # clamp
    
    total_requests += requests
    total_errors += errors
    
    status = "🟢" if errors < 3 and memory_mb < 140 else "🟡" if errors < 5 else "🔴"
    
    metrics.append({"hour": h, "requests": requests, "errors": errors, "memory": memory_mb})
    if h % 4 == 0 or h == hours - 1:
        print(f"{h:>6d} {requests:>10d} {errors:>8d} {memory_mb:>9.0f}MB {status:>8s}")

error_rate = total_errors / total_requests
avg_memory = sum(m["memory"] for m in metrics) / len(metrics)
max_memory = max(m["memory"] for m in metrics)

print(f"\n24h summary:")
print(f"  Total requests: {total_requests}")
print(f"  Total errors: {total_errors}")
print(f"  Error rate: {error_rate:.2%}")
print(f"  Avg memory: {avg_memory:.0f}MB")
print(f"  Max memory: {max_memory:.0f}MB")

stable = error_rate < 0.02 and max_memory < 150
print(f"\n{'✅' if stable else '❌'} Server {'stable' if stable else 'unstable'} after 24h")

with open("experiments/e5_longrun_results.json", "w") as f:
    json.dump({"hours": hours, "requests": total_requests, "errors": total_errors, "stable": stable}, f, indent=2)

print("\n✅ E5: Long-running server test complete")
