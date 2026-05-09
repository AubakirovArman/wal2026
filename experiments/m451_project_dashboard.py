"""
M451 — Project Statistics Dashboard

Aggregates all project metrics into a single view.
"""
import json, os, glob

experiments = len(glob.glob("experiments/m*.py"))
results = len(glob.glob("experiments/*_results.json"))
books = len(glob.glob("book/*.md"))
guides = len(glob.glob("docs/*.md"))

print("=" * 60)
print("M451 — PROJECT DASHBOARD")
print("=" * 60)
print(f"  Experiments:    {experiments}")
print(f"  Result files:   {results}")
print(f"  Book entries:   {books}")
print(f"  Guides:         {guides}")
print(f"  Total scripts:  {experiments + guides}")

with open("experiments/m451_dashboard_results.json", "w") as f:
    json.dump({"experiments": experiments, "results": results, "books": books, "guides": guides, "pass": True}, f, indent=2)

print("\n✅ M451: Project dashboard generated")
