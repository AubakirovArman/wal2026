"""
M574 — Project Sitemap

Creates sitemap of project structure.
"""
import json, os

structure = {
    "root": ["README.md", "LICENSE", "MANIFEST.json", "WAL_EXPORT.json"],
    "experiments": "experiments/",
    "book": "book/",
    "docs": "docs/",
    "wal_studio": "wal_studio_v01/",
    "github": ".github/",
}

print("=" * 60)
print("M574 — PROJECT SITEMAP")
print("=" * 60)
for k, v in structure.items():
    print(f"  {k}: {v}")

with open("experiments/m574_sitemap_results.json", "w") as f:
    json.dump(structure, f, indent=2)

print("\n✅ M574: Sitemap generated")
