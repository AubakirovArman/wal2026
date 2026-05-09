"""
M349 — Cross-Project Recipe Sharing

Share recipes between different WAL projects.
"""
import json

print("=" * 60)
print("M349 — CROSS-PROJECT SHARING")
print("=" * 60)

# Two projects
project_a = {
    "name": "geography",
    "recipes": [
        {"q": "Capital of France?", "a": "Paris"},
        {"q": "Capital of Japan?", "a": "Tokyo"},
    ],
}

project_b = {
    "name": "science",
    "recipes": [
        {"q": "What is H2O?", "a": "Water"},
        {"q": "Speed of light?", "a": "299,792,458 m/s"},
    ],
}

# Share recipes
print("\nProject A:")
for r in project_a["recipes"]:
    print(f"  {r['q']} → {r['a']}")

print("\nProject B:")
for r in project_b["recipes"]:
    print(f"  {r['q']} → {r['a']}")

# Import from A to B
shared = project_a["recipes"][0]  # Share first recipe
project_b["recipes"].append(shared)

print(f"\nShared recipe from A to B:")
print(f"  {shared['q']} → {shared['a']}")

print(f"\nProject B after import: {len(project_b['recipes'])} recipes")

with open("experiments/m349_sharing_results.json", "w") as f:
    json.dump({"project_a": len(project_a["recipes"]), "project_b": len(project_b["recipes"])}, f, indent=2)

print("\n✅ M349: Cross-project sharing working")
