"""
M356 — Token Efficiency

Analyze token usage per fact.
"""
import json

print("=" * 60)
print("M356 — TOKEN EFFICIENCY")
print("=" * 60)

facts = [
    {"q": "X?", "a": "Y"},
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is the detailed history and cultural significance of the capital city of France including its founding date and major historical events?", "a": "Paris was founded in the 3rd century BC"},
]

def count_tokens(text):
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)

print("\nToken usage per fact:")
for i, f in enumerate(facts):
    q_tokens = count_tokens(f["q"])
    a_tokens = count_tokens(f["a"])
    total = q_tokens + a_tokens
    print(f"  Fact {i+1}: {total} tokens (Q: {q_tokens}, A: {a_tokens})")

with open("experiments/m356_token_results.json", "w") as f:
    json.dump({"facts": len(facts)}, f, indent=2)

print("\n✅ M356: Token efficiency analyzed")
