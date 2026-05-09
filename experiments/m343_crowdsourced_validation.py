"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M343 — Crowdsourced Validation

Multiple validators check facts for accuracy.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M343 — CROWDSOURCED VALIDATION")
print("=" * 60)

facts = [
    {"id": 1, "q": "Capital of France?", "a": "Paris"},
    {"id": 2, "q": "Capital of Japan?", "a": "Tokyo"},
    {"id": 3, "q": "Capital of Brazil?", "a": "Brasília"},
]

# Simulate 3 validators
validators = ["validator_A", "validator_B", "validator_C"]

print("\nValidation results:")
for fact in facts:
    votes = []
    for v in validators:
        # Simulate validation (95% accuracy)
        correct = random.random() < 0.95
        vote = "correct" if correct else "incorrect"
        votes.append(vote)
    
    consensus = max(set(votes), key=votes.count)
    confidence = votes.count(consensus) / len(votes)
    
    status = "✅" if consensus == "correct" else "❌"
    print(f"  {status} [{fact['id']}] {fact['q']} → {fact['a']}")
    print(f"      Votes: {votes}, Consensus: {consensus} ({confidence:.0%})")

with open("experiments/m343_crowd_results.json", "w") as f:
    json.dump({"facts": len(facts), "validators": len(validators)}, f, indent=2)

print("\n✅ M343: Crowdsourced validation complete")
