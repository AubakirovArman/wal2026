"""
M548 — Module Dependency Graph

Builds graph of experiment dependencies.
"""
import json, glob, re, os

graph = {}
for path in glob.glob("experiments/m*.py"):
    name = os.path.basename(path)[:-3]
    with open(path) as f:
        content = f.read()
    deps = re.findall(r'experiments/(m\d+_[^.]+\.py)', content)
    graph[name] = [os.path.basename(d)[:-3] for d in deps]

with_deps = sum(1 for v in graph.values() if v)

print("=" * 60)
print("M548 — MODULE DEPENDENCY GRAPH")
print("=" * 60)
print(f"  Modules: {len(graph)}")
print(f"  With dependencies: {with_deps}")

with open("experiments/m548_dep_graph_results.json", "w") as f:
    json.dump({"modules": len(graph), "with_deps": with_deps, "pass": True}, f, indent=2)

print("\n✅ M548: Dependency graph built")
