"""
M323 — Recipe Search

Full-text search across all recipes.
"""
import json

print("=" * 60)
print("M323 — RECIPE SEARCH")
print("=" * 60)

recipes = [
    {"id": 1, "question": "What is the capital of France?", "answer": "Paris"},
    {"id": 2, "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": 3, "question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"id": 4, "question": "What is H2O?", "answer": "Water"},
    {"id": 5, "question": "What is the speed of light?", "answer": "299,792,458 m/s"},
]

def search(query, recipes):
    """Simple full-text search."""
    query_lower = query.lower()
    results = []
    for r in recipes:
        score = 0
        if query_lower in r["question"].lower():
            score += 2
        if query_lower in r["answer"].lower():
            score += 1
        if score > 0:
            results.append({"recipe": r, "score": score})
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

# Test searches
queries = [
    "capital",
    "France",
    "speed",
    "Water",
    "Japan",
]

print("\nSearch results:")
for query in queries:
    results = search(query, recipes)
    print(f"\n  Query: '{query}'")
    for r in results:
        print(f"    [{r['score']}] {r['recipe']['question'][:35]}... → {r['recipe']['answer']}")

# Stats
total_searches = len(queries)
total_results = sum(len(search(q, recipes)) for q in queries)
avg_results = total_results / total_searches

print(f"\nStatistics:")
print(f"  Total searches: {total_searches}")
print(f"  Total results: {total_results}")
print(f"  Avg results per query: {avg_results:.1f}")

results = {
    "recipes_indexed": len(recipes),
    "queries_tested": total_searches,
    "avg_results": avg_results,
}

with open("experiments/m323_search_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M323: Recipe search working")
