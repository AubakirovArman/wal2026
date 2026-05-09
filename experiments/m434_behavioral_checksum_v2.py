"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M434 — Behavioral Checksum v2

Deterministic hash of model responses for regression detection.
"""
import json, hashlib

def behavioral_checksum(responses):
    data = json.dumps(responses, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]

responses_v1 = {"Paris": "France", "Berlin": "Germany", "Madrid": "Spain"}
responses_v2 = {"Paris": "France", "Berlin": "Germany", "Madrid": "Spain"}
responses_v3 = {"Paris": "France", "Berlin": "Germany", "Madrid": "Portugal"}

c1 = behavioral_checksum(responses_v1)
c2 = behavioral_checksum(responses_v2)
c3 = behavioral_checksum(responses_v3)

print("=" * 60)
print("M434 — BEHAVIORAL CHECKSUM V2")
print("=" * 60)
print(f"  V1: {c1}")
print(f"  V2: {c2}")
print(f"  V3: {c3}")
print(f"  V1==V2: {'✅' if c1==c2 else '❌'}")
print(f"  V1!=V3: {'✅' if c1!=c3 else '❌'}")

assert c1 == c2
assert c1 != c3

with open("experiments/m434_checksum_results.json", "w") as f:
    json.dump({"v1": c1, "v2": c2, "v3": c3, "pass": True}, f, indent=2)

print("\n✅ M434: Behavioral checksum v2 working")
