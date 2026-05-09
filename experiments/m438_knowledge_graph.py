"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M438 — Knowledge Graph Integration

Builds graph of facts and their relationships.
"""
import json

facts = [
    {"subject": "Paris", "relation": "capital_of", "object": "France"},
    {"subject": "France", "relation": "in", "object": "Europe"},
    {"subject": "Berlin", "relation": "capital_of", "object": "Germany"},
    {"subject": "Germany", "relation": "in", "object": "Europe"},
]

# Build adjacency
graph = {}
for f in facts:
    s = f["subject"]
    o = f["object"]
    if s not in graph:
        graph[s] = []
    graph[s].append({"relation": f["relation"], "to": o})

print("=" * 60)
print("M438 — KNOWLEDGE GRAPH")
print("=" * 60)

for node, edges in graph.items():
    print(f"  {node}:")
    for e in edges:
        print(f"    → {e['to']} ({e['relation']})")

assert "Paris" in graph
assert len(graph["Paris"]) == 1

with open("experiments/m438_knowledge_graph_results.json", "w") as f:
    json.dump({"nodes": list(graph.keys()), "edges": sum(len(v) for v in graph.values()), "pass": True}, f, indent=2)

print("\n✅ M438: Knowledge graph integration working")
