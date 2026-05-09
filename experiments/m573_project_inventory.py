"""
M573 — Project Inventory

Lists all project assets.
"""
import json, glob, os

assets = {
    "experiments": sorted(glob.glob("experiments/m*.py")),
    "books": sorted(glob.glob("book/*.md")),
    "docs": sorted(glob.glob("docs/**/*.md", recursive=True)),
    "results": sorted(glob.glob("experiments/*_results.json")),
}

print("=" * 60)
print("M573 — PROJECT INVENTORY")
print("=" * 60)
for k, v in assets.items():
    print(f"  {k}: {len(v)}")

with open("experiments/m573_inventory_results.json", "w") as f:
    json.dump({k: len(v) for k, v in assets.items()}, f, indent=2)

print("\n✅ M573: Inventory complete")
