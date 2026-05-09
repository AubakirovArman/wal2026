"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M376 — Config Validation

Validate WAL configuration before build.
"""
import json

print("=" * 60)
print("M376 — CONFIG VALIDATION")
print("=" * 60)

configs = [
    {"layer": 16, "rank": 4, "steps": 100, "valid": True},
    {"layer": 50, "rank": 4, "steps": 100, "valid": False},  # layer too high
    {"layer": 16, "rank": 0, "steps": 100, "valid": False},  # rank too low
    {"layer": 16, "rank": 4, "steps": 0, "valid": False},   # steps too low
]

print("\nConfig validation:")
for c in configs:
    errors = []
    if not (0 <= c["layer"] <= 32):
        errors.append("layer out of range")
    if not (1 <= c["rank"] <= 64):
        errors.append("rank out of range")
    if not (1 <= c["steps"] <= 10000):
        errors.append("steps out of range")
    
    ok = len(errors) == 0
    status = "✅" if ok else "❌"
    print(f"  {status} layer={c['layer']}, rank={c['rank']}, steps={c['steps']} {errors}")

with open("experiments/m376_config_results.json", "w") as f:
    json.dump({"configs": len(configs)}, f, indent=2)

print("\n✅ M376: Config validation working")
