"""
M534 — Module Counter

Counts total modules by prefix.
"""
import json, glob, os

prefixes = {}
for path in glob.glob("experiments/*.py"):
    name = os.path.basename(path)
    prefix = name[:2] if name[0].isalpha() else "other"
    prefixes[prefix] = prefixes.get(prefix, 0) + 1

print("=" * 60)
print("M534 — MODULE COUNTER")
print("=" * 60)
for p, c in sorted(prefixes.items()):
    print(f"  {p}: {c}")

with open("experiments/m534_module_count_results.json", "w") as f:
    json.dump(prefixes, f, indent=2)

print("\n✅ M534: Module counts complete")
