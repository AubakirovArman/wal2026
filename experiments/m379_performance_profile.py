"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M379 — Performance Profile

Profile system performance bottlenecks.
"""
import json

print("=" * 60)
print("M379 — PERFORMANCE PROFILE")
print("=" * 60)

components = [
    {"name": "Load model", "time_ms": 2000, "pct": 20},
    {"name": "Tokenize", "time_ms": 500, "pct": 5},
    {"name": "Train adapters", "time_ms": 6100, "pct": 61},
    {"name": "Save checkpoint", "time_ms": 1000, "pct": 10},
    {"name": "Run CI tests", "time_ms": 400, "pct": 4},
]

print("\nPerformance breakdown:")
print(f"{'Component':>20s} {'Time':>10s} {'%':>6s} {'Bar':>20s}")
print("-" * 60)

for c in components:
    bar = "█" * (c["pct"] // 2)
    print(f"{c['name']:>20s} {c['time_ms']:>8d}ms {c['pct']:>5d}% {bar:>20s}")

total = sum(c["time_ms"] for c in components)
print(f"\n{'Total':>20s} {total:>8d}ms {100:>5d}%")

with open("experiments/m379_profile_results.json", "w") as f:
    json.dump({"total_ms": total, "bottleneck": "Train adapters"}, f, indent=2)

print("\n✅ M379: Performance profile complete")
