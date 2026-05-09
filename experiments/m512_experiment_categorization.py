"""
M512 — Experiment Categorization

Groups experiments by topic.
"""
import json, glob, os, re

categories = {
    "core": [],
    "security": [],
    "infra": [],
    "validation": [],
    "meta": [],
}

for path in glob.glob("experiments/m*.py"):
    name = os.path.basename(path)
    if any(x in name for x in ["security", "injection", "attack"]):
        categories["security"].append(name)
    elif any(x in name for x in ["ci", "test", "valid", "benchmark"]):
        categories["validation"].append(name)
    elif any(x in name for x in ["git", "repo", "github", "doc", "book", "readme", "changelog"]):
        categories["meta"].append(name)
    elif any(x in name for x in ["deploy", "docker", "k8s", "server", "api", "monitor", "alert", "backup", "scale", "load"]):
        categories["infra"].append(name)
    else:
        categories["core"].append(name)

print("=" * 60)
print("M512 — EXPERIMENT CATEGORIZATION")
print("=" * 60)
for cat, items in categories.items():
    print(f"  {cat}: {len(items)}")

with open("experiments/m512_categorization_results.json", "w") as f:
    json.dump({k: len(v) for k, v in categories.items()}, f, indent=2)

print("\n✅ M512: Experiments categorized")
