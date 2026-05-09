"""
M332 — Fact Similarity Matrix

Compute similarity between all facts.
"""
import json

print("=" * 60)
print("M332 — FACT SIMILARITY MATRIX")
print("=" * 60)

facts = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Brazil?",
    "What language is spoken in France?",
    "What is the currency of France?",
    "What is H2O?",
]

def similarity(a, b):
    """Word overlap similarity."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0

# Compute similarity matrix
matrix = []
for i, f1 in enumerate(facts):
    row = []
    for j, f2 in enumerate(facts):
        sim = similarity(f1, f2)
        row.append(sim)
    matrix.append(row)

print("\nSimilarity matrix:")
print(f"{'':>35s}", end="")
for i in range(len(facts)):
    print(f"  F{i+1}", end="")
print()

for i, f1 in enumerate(facts):
    short = f1[:30] + "..."
    print(f"{short:>35s}", end="")
    for j in range(len(facts)):
        print(f"  {matrix[i][j]:.2f}", end="")
    print()

# Find most similar pairs
print("\nMost similar pairs:")
pairs = []
for i in range(len(facts)):
    for j in range(i+1, len(facts)):
        pairs.append((facts[i], facts[j], matrix[i][j]))

pairs.sort(key=lambda x: x[2], reverse=True)
for f1, f2, sim in pairs[:3]:
    print(f"  {sim:.2f}: '{f1[:25]}...' ≈ '{f2[:25]}...'")

with open("experiments/m332_similarity_results.json", "w") as f:
    json.dump({
        "facts": len(facts),
        "max_similarity": max(matrix[i][j] for i in range(len(facts)) for j in range(i+1, len(facts))),
        "avg_similarity": sum(sum(row) for row in matrix) / (len(facts) * len(facts)),
    }, f, indent=2)

print("\n✅ M332: Similarity matrix computed")
