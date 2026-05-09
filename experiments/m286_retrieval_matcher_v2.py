"""
M286 — Retrieval Matcher v2

Hypothesis: Better matching than exact: fuzzy + token overlap + embedding similarity.
"""
import os, sys, json, re

def exact_match(query, fact_q):
    return query.lower().strip("?") == fact_q.lower().strip("?")

def fuzzy_match(query, fact_q, threshold=0.6):
    """Simple token overlap ratio."""
    q_tokens = set(query.lower().split())
    f_tokens = set(fact_q.lower().split())
    if not q_tokens or not f_tokens:
        return False
    overlap = len(q_tokens & f_tokens) / len(q_tokens | f_tokens)
    return overlap >= threshold

def substring_match(query, fact_q):
    return query.lower().strip("?") in fact_q.lower().strip("?") or fact_q.lower().strip("?") in query.lower().strip("?")

RETRIEVAL_DB = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
]

TEST_QUERIES = [
    "What is the capital of France?",  # Exact
    "Capital of France?",  # Shortened
    "France capital city",  # Reordered
    "What is the capital of Japan?",  # Exact
    "Japan's capital",  # Possessive
    "Tell me about Italy's capital",  # Different structure
    "Rome is the capital of which country?",  # Reverse
    "Madrid capital of what",  # Reordered
]

print("=" * 60)
print("M286 — Retrieval Matcher v2")
print("=" * 60)

results = []
for query in TEST_QUERIES:
    # Try exact
    matched = None
    for q, a in RETRIEVAL_DB:
        if exact_match(query, q):
            matched = a
            match_type = "exact"
            break
    
    # Try fuzzy
    if not matched:
        for q, a in RETRIEVAL_DB:
            if fuzzy_match(query, q):
                matched = a
                match_type = "fuzzy"
                break
    
    # Try substring
    if not matched:
        for q, a in RETRIEVAL_DB:
            if substring_match(query, q):
                matched = a
                match_type = "substring"
                break
    
    status = "✅" if matched else "❌"
    print(f"  {status} '{query[:40]}...' → {matched} ({match_type if matched else 'none'})")
    results.append({"query": query, "matched": matched, "type": match_type if matched else None})

matched_count = sum(1 for r in results if r["matched"])

print("\n" + "=" * 60)
print(f"SUMMARY")
print("=" * 60)
print(f"  Matched: {matched_count}/{len(TEST_QUERIES)}")

if matched_count >= len(TEST_QUERIES) * 0.7:
    print("\n🎯 RETRIEVAL MATCHER V2 WORKS")
else:
    print("\n⚠️  Matcher needs improvement")
print("=" * 60)

with open("experiments/m286_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m286_results.json")
