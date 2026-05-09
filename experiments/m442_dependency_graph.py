"""
M442 — Dependency Graph Builder

Tracks which facts depend on others.
"""
import json

facts = [
    {"id": "f1", "fact": "Paris is capital of France"},
    {"id": "f2", "fact": "France is in Europe", "depends_on": "f1"},
    {"id": "f3", "fact": "EU has 27 members", "depends_on": "f2"},
    {"id": "f4", "fact": "Berlin is capital of Germany"},
]

# Build dependency graph
deps = {}
for f in facts:
    fid = f["id"]
    deps[fid] = f.get("depends_on", None)

print("=" * 60)
print("M442 — DEPENDENCY GRAPH")
print("=" * 60)

for fid, dep in deps.items():
    print(f"  {fid}: depends_on={dep}")

# Validate no cycles
def has_cycle(fid, visited=None):
    if visited is None:
        visited = set()
    if fid in visited:
        return True
    if fid is None or fid not in deps or deps[fid] is None:
        return False
    visited.add(fid)
    return has_cycle(deps[fid], visited)

cycle_free = not any(has_cycle(fid) for fid in deps)
print(f"\nCycle-free: {'✅' if cycle_free else '❌'}")
assert cycle_free

with open("experiments/m442_dependency_results.json", "w") as f:
    json.dump({"dependencies": deps, "cycle_free": cycle_free, "pass": True}, f, indent=2)

print("\n✅ M442: Dependency graph builder working")
