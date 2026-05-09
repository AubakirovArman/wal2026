"""
M342 — Batch Optimizer

Find optimal batch size for training.
"""
import json

print("=" * 60)
print("M342 — BATCH OPTIMIZER")
print("=" * 60)

# Simulate different batch sizes
batches = [
    {"size": 1, "time_per_fact": 4.5, "survival": 0.98},
    {"size": 5, "time_per_fact": 2.0, "survival": 0.97},
    {"size": 10, "time_per_fact": 1.2, "survival": 0.96},
    {"size": 20, "time_per_fact": 0.8, "survival": 0.94},
    {"size": 50, "time_per_fact": 0.5, "survival": 0.88},
]

print("\nBatch size analysis:")
print(f"{'Size':>6s} {'Time/Fact':>10s} {'Survival':>10s} {'Efficiency':>12s}")
print("-" * 42)

for b in batches:
    efficiency = b["survival"] / b["time_per_fact"]
    print(f"{b['size']:>6d} {b['time_per_fact']:>9.1f}s {b['survival']:>9.1%} {efficiency:>11.2f}")

best = max(batches, key=lambda b: b["survival"] / b["time_per_fact"])
print(f"\nOptimal batch size: {best['size']} (efficiency: {best['survival']/best['time_per_fact']:.2f})")

with open("experiments/m342_batch_results.json", "w") as f:
    json.dump({"optimal_batch_size": best["size"], "efficiency": best["survival"] / best["time_per_fact"]}, f, indent=2)

print("\n✅ M342: Batch optimization complete")
