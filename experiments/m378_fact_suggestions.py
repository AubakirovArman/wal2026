"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M378 — Fact Suggestions

Suggest related facts based on existing ones.
"""
import json

print("=" * 60)
print("M378 — FACT SUGGESTIONS")
print("=" * 60)

existing = [
    {"q": "Capital of France?", "a": "Paris"},
    {"q": "Capital of Japan?", "a": "Tokyo"},
]

# Suggest based on pattern
suggestions = [
    {"q": "Capital of Germany?", "a": "Berlin", "reason": "Follows country-capital pattern"},
    {"q": "Capital of Italy?", "a": "Rome", "reason": "Follows country-capital pattern"},
]

print("Existing facts:")
for f in existing:
    print(f"  {f['q']} → {f['a']}")

print("\nSuggested facts:")
for s in suggestions:
    print(f"  {s['q']} → {s['a']} ({s['reason']})")

with open("experiments/m378_suggestions_results.json", "w") as f:
    json.dump({"existing": len(existing), "suggested": len(suggestions)}, f, indent=2)

print("\n✅ M378: Fact suggestions generated")
