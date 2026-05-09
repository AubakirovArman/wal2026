"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M427 — Memory Leak Checker v2

Tracks memory growth over time with regression detection.
"""
import json

def detect_leak(samples, threshold_mb_per_hour=2):
    if len(samples) < 3:
        return False, "Not enough samples"
    # Linear regression on last N samples
    n = len(samples)
    x = list(range(n))
    y = samples
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / sum((xi - x_mean) ** 2 for xi in x)
    if slope > threshold_mb_per_hour:
        return True, f"Leak detected: {slope:.1f} MB/hour"
    return False, f"Stable: {slope:.1f} MB/hour"

# Simulated: leaky vs fixed
leaky_samples = [100, 102, 105, 108, 112, 115, 118, 122]
fixed_samples = [100, 101, 101, 102, 102, 103, 103, 103]

print("=" * 60)
print("M427 — MEMORY LEAK CHECKER V2")
print("=" * 60)

leak, msg = detect_leak(leaky_samples)
print(f"  Leaky system: {msg} {'❌' if leak else '✅'}")
assert leak

leak, msg = detect_leak(fixed_samples)
print(f"  Fixed system: {msg} {'❌' if leak else '✅'}")
assert not leak

with open("experiments/m427_leak_checker_results.json", "w") as f:
    json.dump({"leaky_detected": True, "fixed_detected": False, "pass": True}, f, indent=2)

print("\n✅ M427: Memory leak checker v2 working")
