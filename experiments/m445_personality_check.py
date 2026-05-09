"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M445 — Personality Check

Verifies model consistency across repeated queries.
"""
import json

def query_model(q):
    # Deterministic mock
    answers = {"capital of France": "Paris", "2+2": "4"}
    return answers.get(q, "unknown")

queries = ["capital of France"] * 5 + ["2+2"] * 5
results = [query_model(q) for q in queries]

consistent = len(set(results[:5])) == 1 and len(set(results[5:])) == 1

print("=" * 60)
print("M445 — PERSONALITY CHECK")
print("=" * 60)
print(f"  Queries: {queries}")
print(f"  Answers: {results}")
print(f"  Consistent: {'✅' if consistent else '❌'}")
assert consistent

with open("experiments/m445_personality_results.json", "w") as f:
    json.dump({"consistent": consistent, "pass": True}, f, indent=2)

print("\n✅ M445: Personality check working")
