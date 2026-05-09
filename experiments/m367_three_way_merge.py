"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M367 — Three-Way Merge

Resolve edit conflicts with 3-way merge.
"""
import json

print("=" * 60)
print("M367 — THREE-WAY MERGE")
print("=" * 60)

# Base, branch A, branch B
base = {"q": "Capital of France?", "a": "Paris"}
a = {"q": "Capital of France?", "a": "Paris"}  # unchanged
b = {"q": "Capital of France?", "a": "Paris, France"}  # modified

def three_way_merge(base, a, b):
    """Simple 3-way merge."""
    result = {}
    for key in base:
        if a.get(key) == b.get(key):
            result[key] = a[key]  # Both agree
        elif a.get(key) == base.get(key):
            result[key] = b[key]  # B changed
        elif b.get(key) == base.get(key):
            result[key] = a[key]  # A changed
        else:
            result[key] = f"CONFLICT: A={a.get(key)} vs B={b.get(key)}"
    return result

merged = three_way_merge(base, a, b)

print("\n3-way merge:")
print(f"  Base: {base}")
print(f"  A:    {a}")
print(f"  B:    {b}")
print(f"  Merged: {merged}")

with open("experiments/m367_merge_results.json", "w") as f:
    json.dump({"conflict": "CONFLICT" in str(merged)}, f, indent=2)

print("\n✅ M367: Three-way merge working")
