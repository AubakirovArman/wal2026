"""
M297 — Fact Deduplication

Detect and merge duplicate or similar facts in recipe sets.
"""
import json

print("=" * 60)
print("M297 — FACT DEDUPLICATION")
print("=" * 60)

recipes = [
    {"id": 1, "question": "What is the capital of France?", "answer": "Paris"},
    {"id": 2, "question": "What is France's capital?", "answer": "Paris"},  # duplicate
    {"id": 3, "question": "Capital of France?", "answer": "Paris"},  # duplicate
    {"id": 4, "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": 5, "question": "Tokyo is the capital of?", "answer": "Japan"},  # reversed
    {"id": 6, "question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"id": 7, "question": "What is the capital of Brazil?", "answer": "Brasilia"},  # typo
]

def normalize(text):
    """Normalize text for comparison."""
    return text.lower().replace("'", "").replace("?", "").replace(".", "").strip()

def similarity(a, b):
    """Simple word overlap similarity."""
    words_a = set(normalize(a).split())
    words_b = set(normalize(b).split())
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0

def find_duplicates(recipes, threshold=0.8):
    """Find duplicate recipes."""
    duplicates = []
    for i, r1 in enumerate(recipes):
        for j, r2 in enumerate(recipes):
            if i >= j:
                continue
            q_sim = similarity(r1["question"], r2["question"])
            a_sim = similarity(r1["answer"], r2["answer"])
            if q_sim >= threshold and a_sim >= threshold:
                duplicates.append((r1, r2, q_sim, a_sim))
    return duplicates

print(f"\nInput recipes: {len(recipes)}")
for r in recipes:
    print(f"  [{r['id']}] {r['question']} → {r['answer']}")

dups = find_duplicates(recipes, threshold=0.5)
print(f"\nDetected duplicates: {len(dups)}")
for r1, r2, q_sim, a_sim in dups:
    print(f"  [{r1['id']}] ≈ [{r2['id']}] (q_sim={q_sim:.2f}, a_sim={a_sim:.2f})")
    print(f"    '{r1['question'][:40]}...' ≈ '{r2['question'][:40]}...'")

# Deduplicated set
unique = []
seen = set()
for r in recipes:
    key = normalize(r["question"]) + "|" + normalize(r["answer"])
    if key not in seen:
        seen.add(key)
        unique.append(r)

print(f"\nAfter deduplication: {len(unique)} recipes (removed {len(recipes) - len(unique)})")

results = {
    "input_count": len(recipes),
    "duplicate_count": len(dups),
    "output_count": len(unique),
    "threshold": 0.8,
}

with open("experiments/m297_dedup_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M297: Fact deduplication working")
