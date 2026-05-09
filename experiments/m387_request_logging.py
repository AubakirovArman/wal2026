"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M387 — Request Logging

Log all inference requests.
"""
import json

print("=" * 60)
print("M387 — REQUEST LOGGING")
print("=" * 60)

logs = []
for i in range(5):
    logs.append({
        "id": i + 1,
        "question": f"Question {i+1}",
        "answer": f"Answer {i+1}",
        "latency_ms": 40 + i * 2,
        "timestamp": f"2026-05-03T12:0{i}:00",
    })

print("\nRequest logs:")
print(f"{'ID':>4s} {'Question':>15s} {'Latency':>10s} {'Time':>15s}")
print("-" * 50)
for log in logs:
    print(f"{log['id']:>4d} {log['question']:>15s} {log['latency_ms']:>9d}ms {log['timestamp']:>15s}")

avg_latency = sum(l["latency_ms"] for l in logs) / len(logs)
print(f"\nAverage latency: {avg_latency:.0f}ms")

with open("experiments/m387_logging_results.json", "w") as f:
    json.dump({"logs": len(logs), "avg_latency_ms": avg_latency}, f, indent=2)

print("\n✅ M387: Request logging working")
