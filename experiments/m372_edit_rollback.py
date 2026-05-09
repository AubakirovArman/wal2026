"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M372 — Edit Rollback

Rollback specific edits from history.
"""
import json

print("=" * 60)
print("M372 — EDIT ROLLBACK")
print("=" * 60)

history = [
    {"id": 1, "action": "add", "fact": "Paris=France"},
    {"id": 2, "action": "add", "fact": "Tokyo=Japan"},
    {"id": 3, "action": "add", "fact": "Berlin=Germany"},
]

print("Original history:")
for h in history:
    print(f"  [{h['id']}] {h['action']}: {h['fact']}")

# Rollback edit 2
to_rollback = 2
print(f"\nRolling back edit {to_rollback}...")
history = [h for h in history if h["id"] != to_rollback]

print("After rollback:")
for h in history:
    print(f"  [{h['id']}] {h['action']}: {h['fact']}")

with open("experiments/m372_rollback_results.json", "w") as f:
    json.dump({"remaining": len(history)}, f, indent=2)

print("\n✅ M372: Edit rollback working")
