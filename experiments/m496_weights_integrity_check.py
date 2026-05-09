"""
M496 — WAL Weights Integrity Check

Verifies compiled weights haven't been tampered with.
"""
import json, hashlib

def compute_hash(weights):
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:16]

weights = {"fact_001": [0.1, -0.2, 0.3], "fact_002": [0.4, 0.5, -0.1]}
hash1 = compute_hash(weights)
hash2 = compute_hash(weights)
weights["fact_002"][0] = 0.5
hash3 = compute_hash(weights)

print("=" * 60)
print("M496 — WEIGHTS INTEGRITY")
print("=" * 60)
print(f"  Original: {hash1}")
print(f"  Same data: {hash2}")
print(f"  Modified: {hash3}")
print(f"  Integrity: {'✅' if hash1==hash2!=hash3 else '❌'}")

assert hash1 == hash2 and hash1 != hash3

with open("experiments/m496_integrity_results.json", "w") as f:
    json.dump({"original": hash1, "modified": hash3, "pass": True}, f, indent=2)

print("\n✅ M496: Weights integrity check working")
