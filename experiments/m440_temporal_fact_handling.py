"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M440 — Temporal Fact Handling

Handles facts that change over time (e.g., presidents).
"""
import json

facts = [
    {"fact": "US President", "value": "Biden", "valid_from": "2021-01-20", "valid_to": "2025-01-20"},
    {"fact": "US President", "value": "Trump", "valid_from": "2025-01-20", "valid_to": None},
]

def query_fact(facts, fact_name, date):
    for f in facts:
        if f["fact"] == fact_name:
            if f["valid_from"] <= date:
                if f["valid_to"] is None or date < f["valid_to"]:
                    return f["value"]
    return None

print("=" * 60)
print("M440 — TEMPORAL FACT HANDLING")
print("=" * 60)

for date in ["2023-06-01", "2025-06-01"]:
    result = query_fact(facts, "US President", date)
    print(f"  On {date}: US President = {result}")

assert query_fact(facts, "US President", "2023-06-01") == "Biden"
assert query_fact(facts, "US President", "2025-06-01") == "Trump"

with open("experiments/m440_temporal_results.json", "w") as f:
    json.dump({"queries": 2, "correct": 2, "pass": True}, f, indent=2)

print("\n✅ M440: Temporal fact handling working")
