"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M416 — Smart Rehearsal System

Selects facts for rehearsal based on forgetting curve.
"""
import json, math

def forgetting_curve(days, stability=5):
    return math.exp(-days / stability)

facts = [
    {"id": "f1", "last_seen": 0, "stability": 5},
    {"id": "f2", "last_seen": 2, "stability": 3},
    {"id": "f3", "last_seen": 5, "stability": 10},
    {"id": "f4", "last_seen": 1, "stability": 2},
    {"id": "f5", "last_seen": 10, "stability": 7},
]

print("=" * 60)
print("M416 — SMART REHEARSAL SYSTEM")
print("=" * 60)

for f in facts:
    retention = forgetting_curve(f["last_seen"], f["stability"])
    f["retention"] = round(retention, 3)
    print(f"  {f['id']}: last_seen={f['last_seen']}d, stability={f['stability']}, retention={f['retention']}")

# Select facts with retention < 0.5 for rehearsal
rehearse = [f for f in facts if f["retention"] < 0.5]
print(f"\nSelected for rehearsal: {[f['id'] for f in rehearse]}")

assert len(rehearse) >= 1
with open("experiments/m416_rehearsal_results.json", "w") as f:
    json.dump({"facts": facts, "rehearse": [r["id"] for r in rehearse], "pass": True}, f, indent=2)

print("\n✅ M416: Smart rehearsal system working")
