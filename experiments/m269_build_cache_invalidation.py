"""
M269 — Build Cache Invalidation Rules

Hypothesis: We can define precise rules for when a cached build
must be invalidated and rebuilt.
"""
import os, sys, json, hashlib, time

# Define what invalidates a cached build
INVALIDATION_RULES = {
    "seed_change": "Recipe seed changed → invalidate",
    "K_change": "Hadamard K changed → invalidate",
    "layer_change": "Target layer changed → invalidate",
    "module_change": "Target modules changed → invalidate",
    "recipe_hash_change": "Recipe content changed → invalidate",
    "dataset_hash_change": "Training dataset changed → invalidate",
    "code_version_change": "WAL code version changed → invalidate",
}

def compute_recipe_hash(recipes):
    h = hashlib.sha256()
    for r in sorted(recipes, key=lambda x: x.get("id", 0)):
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()[:16]

def should_invalidate(cache_entry, new_config):
    """Check if cache should be invalidated."""
    reasons = []
    
    if cache_entry.get("seed") != new_config.get("seed"):
        reasons.append("seed_change")
    if cache_entry.get("K") != new_config.get("K"):
        reasons.append("K_change")
    if cache_entry.get("layer") != new_config.get("layer"):
        reasons.append("layer_change")
    if set(cache_entry.get("modules", [])) != set(new_config.get("modules", [])):
        reasons.append("module_change")
    if cache_entry.get("recipe_hash") != new_config.get("recipe_hash"):
        reasons.append("recipe_hash_change")
    if cache_entry.get("dataset_hash") != new_config.get("dataset_hash"):
        reasons.append("dataset_hash_change")
    if cache_entry.get("code_version") != new_config.get("code_version"):
        reasons.append("code_version_change")
    
    return len(reasons) > 0, reasons

print("=" * 60)
print("M269 — Build Cache Invalidation Rules")
print("=" * 60)

# Test scenarios
scenarios = [
    {
        "name": "No change",
        "cache": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "new": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "expected_invalidate": False,
    },
    {
        "name": "Seed changed",
        "cache": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "new": {"seed": 43, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "expected_invalidate": True,
    },
    {
        "name": "K changed",
        "cache": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "new": {"seed": 42, "K": 512, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "expected_invalidate": True,
    },
    {
        "name": "Layer changed",
        "cache": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "new": {"seed": 42, "K": 256, "layer": 17, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "expected_invalidate": True,
    },
    {
        "name": "Recipe added",
        "cache": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "abc", "code_version": "1.0"},
        "new": {"seed": 42, "K": 256, "layer": 16, "modules": ["q", "v"], "recipe_hash": "def", "code_version": "1.0"},
        "expected_invalidate": True,
    },
]

results = []
for s in scenarios:
    invalidate, reasons = should_invalidate(s["cache"], s["new"])
    correct = invalidate == s["expected_invalidate"]
    status = "✅" if correct else "❌"
    print(f"  {status} {s['name']:<20s} invalidate={invalidate} reasons={reasons}")
    results.append({"name": s["name"], "correct": correct, "reasons": reasons})

correct_count = sum(1 for r in results if r["correct"])

print("\n" + "=" * 60)
print(f"SUMMARY")
print("=" * 60)
print(f"  Correct: {correct_count}/{len(scenarios)}")

if correct_count == len(scenarios):
    print("\n🎯 CACHE INVALIDATION RULES WORK")
else:
    print("\n⚠️  Some rules need fixing")
print("=" * 60)

with open("experiments/m269_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m269_results.json")
