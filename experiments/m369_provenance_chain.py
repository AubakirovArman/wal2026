"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M369 — Provenance Chain

Full history of each fact.
"""
import json

print("=" * 60)
print("M369 — PROVENANCE CHAIN")
print("=" * 60)

# Fact with full provenance
fact = {
    "id": 1,
    "current": {"q": "Capital of France?", "a": "Paris"},
    "history": [
        {"action": "create", "user": "alice", "date": "2026-05-01", "value": "Paris"},
        {"action": "verify", "user": "bob", "date": "2026-05-02", "value": "Paris"},
        {"action": "update", "user": "carol", "date": "2026-05-03", "value": "Paris"},
    ],
}

print("\nProvenance chain:")
print(f"  Fact: {fact['current']['q']} → {fact['current']['a']}")
print(f"  History ({len(fact['history'])} events):")

for h in fact["history"]:
    print(f"    [{h['date']}] {h['action']} by {h['user']}: {h['value']}")

with open("experiments/m369_provenance_results.json", "w") as f:
    json.dump({"fact_id": fact["id"], "history_length": len(fact["history"])}, f, indent=2)

print("\n✅ M369: Provenance chain complete")
