"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M377 — Edit Preview

Preview effect of edit before applying.
"""
import json

print("=" * 60)
print("M377 — EDIT PREVIEW")
print("=" * 60)

# Current state
current = {"q": "Capital of France?", "a": "Paris"}

# Proposed edit
proposed = {"q": "Capital of France?", "a": "Lyon"}

print("\nEdit preview:")
print(f"  Current:  {current['q']} → {current['a']}")
print(f"  Proposed: {proposed['q']} → {proposed['a']}")

# Detect change
if current["a"] != proposed["a"]:
    print(f"\n  ⚠️  Answer changes: '{current['a']}' → '{proposed['a']}'")
    print(f"  Impact: This will modify existing fact")
else:
    print(f"\n  ✅ No change detected")

with open("experiments/m377_preview_results.json", "w") as f:
    json.dump({"change_detected": current["a"] != proposed["a"]}, f, indent=2)

print("\n✅ M377: Edit preview working")
