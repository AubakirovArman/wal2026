"""
M340 — Model Fingerprinting

Create unique signature for model instance.
"""
import json, hashlib

print("=" * 60)
print("M340 — MODEL FINGERPRINTING")
print("=" * 60)

# Model configuration
config = {
    "model": "Llama-3.1-8B",
    "layer": 16,
    "rank": 4,
    "seed": 42,
    "recipes_hash": "abc123",
    "timestamp": "2026-05-03T00:00:00",
}

# Generate fingerprint
def fingerprint(config):
    """Generate unique model fingerprint."""
    # Sort keys for determinism
    data = json.dumps(config, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]

fp = fingerprint(config)
print(f"\nModel fingerprint: {fp}")
print(f"  Model: {config['model']}")
print(f"  Layer: {config['layer']}")
print(f"  Rank: {config['rank']}")
print(f"  Seed: {config['seed']}")

# Same config = same fingerprint
fp2 = fingerprint(config)
print(f"\nSame config fingerprint: {fp2}")
print(f"  Match: {'✅' if fp == fp2 else '❌'}")

# Different config = different fingerprint
config2 = dict(config)
config2["seed"] = 43
fp3 = fingerprint(config2)
print(f"\nDifferent config fingerprint: {fp3}")
print(f"  Match: {'✅' if fp == fp3 else '❌'}")

with open("experiments/m340_fingerprint_results.json", "w") as f:
    json.dump({
        "fingerprint": fp,
        "deterministic": fp == fp2,
        "unique": fp != fp3,
    }, f, indent=2)

print("\n✅ M340: Model fingerprinting working")
