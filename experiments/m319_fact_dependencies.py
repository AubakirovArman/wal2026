"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M319 — Fact Dependencies

Facts that depend on other facts.
"""
import json

print("=" * 60)
print("M319 — FACT DEPENDENCIES")
print("=" * 60)

# Facts with dependencies
facts = {
    "fact_1": {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "depends_on": [],
    },
    "fact_2": {
        "question": "What language is spoken in Paris?",
        "answer": "French",
        "depends_on": ["fact_1"],
    },
    "fact_3": {
        "question": "What is the currency of France?",
        "answer": "Euro",
        "depends_on": ["fact_1"],
    },
    "fact_4": {
        "question": "What is the Eiffel Tower located in?",
        "answer": "Paris",
        "depends_on": ["fact_1"],
    },
    "fact_5": {
        "question": "What country is Paris the capital of?",
        "answer": "France",
        "depends_on": ["fact_1"],
    },
}

def get_dependency_chain(facts, fact_id, chain=None):
    """Get all dependencies for a fact."""
    if chain is None:
        chain = []
    fact = facts.get(fact_id)
    if not fact:
        return chain
    for dep in fact.get("depends_on", []):
        if dep not in chain:
            chain.append(dep)
            get_dependency_chain(facts, dep, chain)
    return chain

def topological_sort(facts):
    """Sort facts by dependencies."""
    sorted_facts = []
    visited = set()
    
    def visit(fact_id):
        if fact_id in visited:
            return
        visited.add(fact_id)
        for dep in facts[fact_id].get("depends_on", []):
            visit(dep)
        sorted_facts.append(fact_id)
    
    for fact_id in facts:
        visit(fact_id)
    
    return sorted_facts

print("\nFact dependencies:")
for fid, fact in facts.items():
    deps = ", ".join(fact["depends_on"]) if fact["depends_on"] else "none"
    print(f"  {fid}: depends on [{deps}]")

print("\nDependency chains:")
for fid in facts:
    chain = get_dependency_chain(facts, fid)
    if chain:
        print(f"  {fid}: {' → '.join(chain + [fid])}")
    else:
        print(f"  {fid}: (root fact)")

print("\nTopological order:")
order = topological_sort(facts)
for i, fid in enumerate(order):
    print(f"  {i+1}. {fid}: {facts[fid]['question'][:35]}...")

# Verify: all dependencies come before dependent facts
valid = True
for i, fid in enumerate(order):
    for dep in facts[fid].get("depends_on", []):
        if dep not in order[:i]:
            valid = False
            print(f"  ❌ {fid} depends on {dep} but comes first!")

print(f"\n{'✅' if valid else '❌'} Topological order valid")

results = {
    "facts": len(facts),
    "dependencies": sum(len(f["depends_on"]) for f in facts.values()),
    "topological_order": order,
    "valid": valid,
}

with open("experiments/m319_dependency_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M319: Fact dependencies with topological sort")
