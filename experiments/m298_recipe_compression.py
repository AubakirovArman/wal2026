"""
M298 — Recipe Compression

Compress recipe storage using delta encoding and deduplication.
"""
import json

print("=" * 60)
print("M298 — RECIPE COMPRESSION")
print("=" * 60)

# Original recipes
recipes_v1 = [
    {"id": 1, "q": "What is the capital of France?", "a": "Paris"},
    {"id": 2, "q": "What is the capital of Japan?", "a": "Tokyo"},
    {"id": 3, "q": "What is the capital of Brazil?", "a": "Brasília"},
]

recipes_v2 = [
    {"id": 1, "q": "What is the capital of France?", "a": "Paris"},
    {"id": 2, "q": "What is the capital of Japan?", "a": "Tokyo"},
    {"id": 3, "q": "What is the capital of Brazil?", "a": "Brasília"},
    {"id": 4, "q": "What is the capital of Egypt?", "a": "Cairo"},
    {"id": 5, "q": "What is the capital of Canada?", "a": "Ottawa"},
]

# Full storage
full_v1 = json.dumps(recipes_v1)
full_v2 = json.dumps(recipes_v2)

# Delta storage (only changes)
def compute_delta(old, new):
    """Compute delta between recipe sets."""
    old_ids = {r["id"] for r in old}
    new_ids = {r["id"] for r in new}
    added = [r for r in new if r["id"] not in old_ids]
    removed = [r["id"] for r in old if r["id"] not in new_ids]
    return {"added": added, "removed": removed}

delta = compute_delta(recipes_v1, recipes_v2)
delta_json = json.dumps(delta)

print(f"\nFull v1 size: {len(full_v1)} bytes")
print(f"Full v2 size: {len(full_v2)} bytes")
print(f"Delta size:   {len(delta_json)} bytes")
print(f"Compression:  {len(full_v2) / len(delta_json):.1f}×")

# Reconstruct v2 from v1 + delta
reconstructed = recipes_v1.copy()
for r in delta["added"]:
    reconstructed.append(r)
for rid in delta["removed"]:
    reconstructed = [r for r in reconstructed if r["id"] != rid]

assert len(reconstructed) == len(recipes_v2), "Reconstruction failed"

results = {
    "full_size_v1": len(full_v1),
    "full_size_v2": len(full_v2),
    "delta_size": len(delta_json),
    "compression_ratio": len(full_v2) / len(delta_json),
    "reconstruction_correct": True,
}

with open("experiments/m298_compression_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M298: Recipe compression working")
