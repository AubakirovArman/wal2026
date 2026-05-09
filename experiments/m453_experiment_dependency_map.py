"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M453 — Experiment Dependency Map

Maps which experiments depend on others.
"""
import json, re, glob, os

deps = {}
for path in glob.glob("experiments/m*.py"):
    with open(path) as f:
        content = f.read()
    name = os.path.basename(path)
    # Find references to other experiments
    refs = re.findall(r'm(\d{3})', content)
    own_num = re.search(r'm(\d{3})', name)
    if own_num:
        own = own_num.group(1)
        deps[name] = [f"m{r}.py" for r in set(refs) if r != own]

print("=" * 60)
print("M453 — EXPERIMENT DEPENDENCY MAP")
print("=" * 60)

# Show top 5 by dependency count
top = sorted(deps.items(), key=lambda x: len(x[1]), reverse=True)[:5]
for name, refs in top:
    print(f"  {name}: {len(refs)} dependencies")

with open("experiments/m453_dependency_map_results.json", "w") as f:
    json.dump({"experiments": len(deps), "max_deps": len(top[0][1]) if top else 0, "pass": True}, f, indent=2)

print("\n✅ M453: Dependency map generated")
