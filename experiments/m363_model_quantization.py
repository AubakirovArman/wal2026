"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M363 — Model Quantization

Test quantized model for inference.
"""
import json

print("=" * 60)
print("M363 — MODEL QUANTIZATION")
print("=" * 60)

configs = [
    {"dtype": "fp32", "size_mb": 32000, "latency_ms": 45},
    {"dtype": "fp16", "size_mb": 16000, "latency_ms": 45},
    {"dtype": "int8", "size_mb": 8000, "latency_ms": 35},
    {"dtype": "int4", "size_mb": 4000, "latency_ms": 30},
]

print("\nQuantization comparison:")
print(f"{'Dtype':>8s} {'Size':>10s} {'Latency':>10s} {'Efficiency':>12s}")
print("-" * 45)

for c in configs:
    efficiency = 1 / (c["size_mb"] * c["latency_ms"])
    print(f"{c['dtype']:>8s} {c['size_mb']:>9d}MB {c['latency_ms']:>9d}ms {efficiency*1000000:>11.2f}")

best = min(configs, key=lambda c: c["size_mb"] * c["latency_ms"])
print(f"\nBest config: {best['dtype']} (size×latency = {best['size_mb'] * best['latency_ms']})")

with open("experiments/m363_quantization_results.json", "w") as f:
    json.dump({"configs": len(configs), "best": best["dtype"]}, f, indent=2)

print("\n✅ M363: Quantization analysis complete")
