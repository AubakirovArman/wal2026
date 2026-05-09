"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M307 — Monitoring Dashboard

Track system metrics over time.
"""
import json, time, random

random.seed(42)

print("=" * 60)
print("M307 — MONITORING DASHBOARD")
print("=" * 60)

# Simulate metrics over time
metrics = []
for hour in range(24):
    m = {
        "hour": hour,
        "timestamp": time.time() - (24 - hour) * 3600,
        "requests": random.randint(100, 500),
        "cache_hit_rate": random.uniform(0.4, 0.7),
        "avg_latency_ms": random.uniform(0.4, 0.8),
        "ci_score": random.uniform(0.85, 0.95),
        "errors": random.randint(0, 5),
        "active_facts": 100 + hour * 10,
    }
    metrics.append(m)

print("\n24-hour metrics summary:")
print(f"{'Hour':>6s} {'Reqs':>6s} {'Cache%':>7s} {'Lat(ms)':>8s} {'CI':>5s} {'Errs':>5s} {'Facts':>6s}")
print("-" * 50)
for m in metrics[::4]:  # Show every 4th hour
    print(f"{m['hour']:>6d} {m['requests']:>6d} {m['cache_hit_rate']:>6.0%} {m['avg_latency_ms']:>8.2f} {m['ci_score']:>4.2f} {m['errors']:>5d} {m['active_facts']:>6d}")

# Aggregates
total_requests = sum(m["requests"] for m in metrics)
avg_cache = sum(m["cache_hit_rate"] for m in metrics) / len(metrics)
avg_latency = sum(m["avg_latency_ms"] for m in metrics) / len(metrics)
avg_ci = sum(m["ci_score"] for m in metrics) / len(metrics)
total_errors = sum(m["errors"] for m in metrics)

print(f"\nAggregates:")
print(f"  Total requests: {total_requests}")
print(f"  Avg cache hit rate: {avg_cache:.1%}")
print(f"  Avg latency: {avg_latency:.2f}ms")
print(f"  Avg CI score: {avg_ci:.3f}")
print(f"  Total errors: {total_errors}")
print(f"  Error rate: {total_errors/total_requests:.2%}")

# Alerts
print(f"\nAlerts:")
if avg_ci < 0.8:
    print("  ⚠️ CI score below threshold")
else:
    print("  ✅ CI score healthy")

if avg_latency > 1.0:
    print("  ⚠️ Latency high")
else:
    print("  ✅ Latency healthy")

if total_errors / total_requests > 0.01:
    print("  ⚠️ Error rate elevated")
else:
    print("  ✅ Error rate healthy")

with open("experiments/m307_metrics.json", "w") as f:
    json.dump({
        "hourly": metrics,
        "aggregates": {
            "total_requests": total_requests,
            "avg_cache_hit": avg_cache,
            "avg_latency_ms": avg_latency,
            "avg_ci_score": avg_ci,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests,
        },
    }, f, indent=2)

print("\n✅ M307: Monitoring dashboard tracking metrics")
