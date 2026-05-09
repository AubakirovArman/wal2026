"""
M380 — Migration Tool

Migrate recipes between WAL versions.
"""
import json

print("=" * 60)
print("M380 — MIGRATION TOOL")
print("=" * 60)

# Old format
old_recipes = [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"},
]

# Migrate to new format
new_recipes = []
for i, r in enumerate(old_recipes):
    new_recipes.append({
        "id": i + 1,
        "version": 1,
        "question": r["question"],
        "answer": r["answer"],
        "created": "2026-05-03",
    })

print("Migration:")
print(f"  Old format: {len(old_recipes)} recipes")
print(f"  New format: {len(new_recipes)} recipes")

for r in new_recipes:
    print(f"    [{r['id']}] v{r['version']} {r['question']} → {r['answer']}")

with open("experiments/m380_migration_results.json", "w") as f:
    json.dump({"migrated": len(new_recipes)}, f, indent=2)

print("\n✅ M380: Migration complete")
