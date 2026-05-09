"""
M509 — Size Analyzer

Analyzes disk usage of project components.
"""
import json, os, glob

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, files in os.walk(path) for f in files)

components = {
    "experiments": "experiments",
    "book": "book",
    "docs": "docs",
    "wal_studio": "wal_studio_v01",
}

print("=" * 60)
print("M509 — SIZE ANALYZER")
print("=" * 60)

sizes = {}
for name, path in components.items():
    if os.path.exists(path):
        size_mb = get_size(path) / 1024**2
        sizes[name] = round(size_mb, 1)
        print(f"  {name}: {size_mb:.1f} MB")

with open("experiments/m509_size_results.json", "w") as f:
    json.dump({"sizes_mb": sizes, "pass": True}, f, indent=2)

print("\n✅ M509: Size analysis complete")
