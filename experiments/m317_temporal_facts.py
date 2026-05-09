"""
M317 — Temporal Facts

Facts that have expiration dates or versioned updates.
"""
import json, time

print("=" * 60)
print("M317 — TEMPORAL FACTS")
print("=" * 60)

# Facts with timestamps
temporal_facts = [
    {
        "id": 1,
        "question": "Who is the current president of USA?",
        "answer": "Joe Biden",
        "valid_from": "2021-01-20",
        "valid_until": "2025-01-20",
        "version": 1,
    },
    {
        "id": 2,
        "question": "Who is the current president of USA?",
        "answer": "Donald Trump",
        "valid_from": "2025-01-20",
        "valid_until": None,  # Current
        "version": 2,
    },
    {
        "id": 3,
        "question": "What is the current year?",
        "answer": "2026",
        "valid_from": "2026-01-01",
        "valid_until": "2026-12-31",
        "version": 1,
    },
]

def get_current_answer(facts, question, current_date=None):
    """Get the current answer for a question."""
    if current_date is None:
        current_date = time.strftime("%Y-%m-%d")
    
    matching = [f for f in facts if f["question"] == question]
    if not matching:
        return None
    
    # Find the most recent valid fact
    current = None
    for f in matching:
        if f["valid_from"] <= current_date:
            if f["valid_until"] is None or f["valid_until"] >= current_date:
                if current is None or f["version"] > current["version"]:
                    current = f
    
    return current

print("\nTemporal fact queries:")
test_dates = ["2024-06-01", "2025-06-01", "2026-06-01"]
for date in test_dates:
    answer = get_current_answer(temporal_facts, "Who is the current president of USA?", date)
    if answer:
        print(f"  [{date}] {answer['answer']} (v{answer['version']})")
    else:
        print(f"  [{date}] No valid answer")

# Fact expiration check
print("\nFact expiration check:")
for f in temporal_facts:
    expired = f["valid_until"] is not None and f["valid_until"] < "2026-05-03"
    status = "EXPIRED" if expired else "ACTIVE"
    print(f"  [{status}] v{f['version']}: {f['answer']} (until {f['valid_until'] or 'now'})")

results = {
    "temporal_facts": len(temporal_facts),
    "active": sum(1 for f in temporal_facts if f["valid_until"] is None or f["valid_until"] >= "2026-05-03"),
    "expired": sum(1 for f in temporal_facts if f["valid_until"] is not None and f["valid_until"] < "2026-05-03"),
}

with open("experiments/m317_temporal_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M317: Temporal facts with versioning")
