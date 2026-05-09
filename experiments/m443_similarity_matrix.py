"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M443 — Similarity Matrix Generator

Computes cosine similarity between fact embeddings.
"""
import json, math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

embeddings = {
    "Paris": [1.0, 0.8, 0.2],
    "Berlin": [0.9, 0.7, 0.3],
    "H2O": [0.2, 0.1, 0.9],
}

print("=" * 60)
print("M443 — SIMILARITY MATRIX")
print("=" * 60)

keys = list(embeddings.keys())
matrix = {}
for i, a in enumerate(keys):
    for b in keys[i:]:
        sim = cosine(embeddings[a], embeddings[b])
        matrix[f"{a}-{b}"] = round(sim, 3)
        print(f"  {a} ↔ {b}: {sim:.3f}")

# Paris and Berlin should be more similar than Paris and H2O
assert matrix["Paris-Berlin"] > matrix["Paris-H2O"]

with open("experiments/m443_similarity_results.json", "w") as f:
    json.dump({"matrix": matrix, "pass": True}, f, indent=2)

print("\n✅ M443: Similarity matrix generator working")
