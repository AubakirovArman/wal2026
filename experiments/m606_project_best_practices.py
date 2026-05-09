"""
M606 — Best Practices

Best practices established during development.
"""
import json

practices = [
    "Every experiment produces _results.json",
    "Use assert for validation",
    "Print final status with ✅",
    "Update dev diary for each batch",
    "Tag releases with git tags",
]

with open("BEST_PRACTICES.md", "w") as f:
    f.write("# Best Practices\n\n")
    for i, p in enumerate(practices, 1):
        f.write(f"{i}. {p}\n")

print("=" * 60)
print("M606 — BEST PRACTICES")
print("=" * 60)
for i, p in enumerate(practices, 1):
    print(f"  {i}. {p}")

with open("experiments/m606_best_practices_results.json", "w") as f:
    json.dump({"practices": len(practices), "pass": True}, f, indent=2)

print("\n✅ M606: Best practices documented")
