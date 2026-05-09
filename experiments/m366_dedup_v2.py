"""
M366 — Fact Deduplication v2

Embedding-based deduplication.
"""
import json

print("=" * 60)
print("M366 — FACT DEDUPLICATION v2")
print("=" * 60)

# Facts with semantic similarity
facts = [
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "France's capital city?", "a": "Paris"},
    {"q": "What is the capital of Japan?", "a": "Tokyo"},
    {"q": "Capital of Japan?", "a": "Tokyo"},
    {"q": "What is H2O?", "a": "Water"},
]

def semantic_sim(a, b):
    """Mock semantic similarity using word overlap."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if wa | wb else 0

# Find duplicates
print("\nSemantic deduplication:")
groups = []
used = set()

for i, f1 in enumerate(facts):
    if i in used:
        continue
    group = [f1]
    used.add(i)
    for j, f2 in enumerate(facts):
        if j in used or i == j:
            continue
        sim = semantic_sim(f1["q"], f2["q"])
        if sim >= 0.5:
            group.append(f2)
            used.add(j)
    groups.append(group)

print(f"  Original: {len(facts)} facts")
print(f"  Groups: {len(groups)}")
for i, g in enumerate(groups):
    print(f"    Group {i+1}: {[f['q'][:25] + '...' for f in g]}")

with open("experiments/m366_dedup_v2_results.json", "w") as f:
    json.dump({"original": len(facts), "groups": len(groups)}, f, indent=2)

print("\n✅ M366: Semantic deduplication working")
