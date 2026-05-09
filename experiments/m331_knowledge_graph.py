"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M331 — Knowledge Graph

Visualize relationships between facts.
"""
import json

print("=" * 60)
print("M331 — KNOWLEDGE GRAPH")
print("=" * 60)

# Facts with relationships
facts = {
    "France": {"capital": "Paris", "language": "French", "currency": "Euro"},
    "Paris": {"country": "France", "landmark": "Eiffel Tower"},
    "Japan": {"capital": "Tokyo", "language": "Japanese"},
    "Tokyo": {"country": "Japan"},
    "Eiffel Tower": {"location": "Paris"},
}

# Build graph
graph = {"nodes": [], "edges": []}
seen = set()

for entity, relations in facts.items():
    if entity not in seen:
        graph["nodes"].append({"id": entity, "type": "entity"})
        seen.add(entity)
    
    for rel, target in relations.items():
        if target not in seen:
            graph["nodes"].append({"id": target, "type": "value"})
            seen.add(target)
        graph["edges"].append({"from": entity, "to": target, "relation": rel})

print(f"\nKnowledge graph:")
print(f"  Nodes: {len(graph['nodes'])}")
print(f"  Edges: {len(graph['edges'])}")

print(f"\nEntities:")
for node in graph["nodes"]:
    print(f"  [{node['type']}] {node['id']}")

print(f"\nRelationships:")
for edge in graph["edges"]:
    print(f"  {edge['from']} --[{edge['relation']}]--> {edge['to']}")

# Find connected components
components = []
visited = set()
for node in graph["nodes"]:
    if node["id"] not in visited:
        component = []
        stack = [node["id"]]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                for edge in graph["edges"]:
                    if edge["from"] == current and edge["to"] not in visited:
                        stack.append(edge["to"])
                    if edge["to"] == current and edge["from"] not in visited:
                        stack.append(edge["from"])
        components.append(component)

print(f"\nConnected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  Component {i+1}: {', '.join(comp)}")

with open("experiments/m331_graph_results.json", "w") as f:
    json.dump({
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "components": len(components),
    }, f, indent=2)

print("\n✅ M331: Knowledge graph generated")
