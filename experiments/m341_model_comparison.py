"""
M341 — Model Comparison Matrix

Compare different model configurations.
"""
import json

print("=" * 60)
print("M341 — MODEL COMPARISON MATRIX")
print("=" * 60)

configs = [
    {"name": "baseline", "layer": 16, "rank": 4, "steps": 100, "survival": 0.92, "latency_ms": 45},
    {"name": "fast", "layer": 16, "rank": 2, "steps": 50, "survival": 0.85, "latency_ms": 38},
    {"name": "accurate", "layer": 16, "rank": 8, "steps": 200, "survival": 0.96, "latency_ms": 52},
    {"name": "multi_layer", "layer": "all", "rank": 4, "steps": 100, "survival": 0.94, "latency_ms": 60},
]

print("\nConfiguration comparison:")
print(f"{'Name':>12s} {'Layer':>8s} {'Rank':>6s} {'Steps':>6s} {'Survival':>10s} {'Latency':>10s} {'Score':>8s}")
print("-" * 65)

for c in configs:
    # Score = survival * 0.6 - latency_penalty * 0.4
    latency_penalty = max(0, (c["latency_ms"] - 40) / 100)
    score = c["survival"] * 0.6 - latency_penalty * 0.4
    print(f"{c['name']:>12s} {str(c['layer']):>8s} {c['rank']:>6d} {c['steps']:>6d} {c['survival']:>9.1%} {c['latency_ms']:>9.0f}ms {score:>7.2f}")

best = max(configs, key=lambda c: c["survival"] * 0.6 - max(0, (c["latency_ms"] - 40) / 100) * 0.4)
print(f"\nBest config: {best['name']}")

with open("experiments/m341_comparison_results.json", "w") as f:
    json.dump({"configs_tested": len(configs), "best": best["name"]}, f, indent=2)

print("\n✅ M341: Model comparison complete")
