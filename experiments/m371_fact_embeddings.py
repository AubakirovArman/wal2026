"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M371 — Fact Embeddings

Generate embeddings for facts (mock).
"""
import json, random

random.seed(42)

print("=" * 60)
print("M371 — FACT EMBEDDINGS")
print("=" * 60)

facts = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is H2O?",
]

print("\nFact embeddings (mock 4D vectors):")
embeddings = {}
for f in facts:
    emb = [round(random.uniform(-1, 1), 2) for _ in range(4)]
    embeddings[f] = emb
    print(f"  '{f[:30]}...' → {emb}")

# Similarity
def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

print("\nPairwise similarities:")
for i, f1 in enumerate(facts):
    for j, f2 in enumerate(facts):
        if i < j:
            sim = cosine_sim(embeddings[f1], embeddings[f2])
            print(f"  {sim:+.2f}: '{f1[:20]}...' ≈ '{f2[:20]}...'")

with open("experiments/m371_embedding_results.json", "w") as f:
    json.dump({"facts": len(facts), "dim": 4}, f, indent=2)

print("\n✅ M371: Fact embeddings generated")
