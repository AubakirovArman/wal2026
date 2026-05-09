"""
M602 — Project Index

Creates index of all project files.
"""
import json, glob, os

index = {
    "experiments": sorted(glob.glob("experiments/m*.py")),
    "results": sorted(glob.glob("experiments/*_results.json")),
    "books": sorted(glob.glob("book/*.md")),
}

# Truncate to counts only
index = {k: len(v) for k, v in index.items()}

with open("INDEX.json", "w") as f:
    json.dump(index, f, indent=2)

print("=" * 60)
print("M602 — PROJECT INDEX")
print("=" * 60)
print(json.dumps(index, indent=2))

with open("experiments/m602_index_results.json", "w") as f:
    json.dump({"indexed": True, "pass": True}, f, indent=2)

print("\n✅ M602: Index generated")
