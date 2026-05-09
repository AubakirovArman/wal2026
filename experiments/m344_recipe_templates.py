"""
M344 — Recipe Templates

Reusable templates for common fact patterns.
"""
import json

print("=" * 60)
print("M344 — RECIPE TEMPLATES")
print("=" * 60)

# Define templates
templates = {
    "capital": {
        "pattern": "What is the capital of {country}?",
        "answer": "{capital}",
    },
    "language": {
        "pattern": "What language is spoken in {country}?",
        "answer": "{language}",
    },
    "currency": {
        "pattern": "What is the currency of {country}?",
        "answer": "{currency}",
    },
}

# Data to fill templates
data = [
    {"country": "France", "capital": "Paris", "language": "French", "currency": "Euro"},
    {"country": "Japan", "capital": "Tokyo", "language": "Japanese", "currency": "Yen"},
]

# Generate recipes from templates
recipes = []
for d in data:
    for name, template in templates.items():
        q = template["pattern"].format(**d)
        a = template["answer"].format(**d)
        recipes.append({"template": name, "question": q, "answer": a})

print(f"\nGenerated {len(recipes)} recipes from {len(templates)} templates:")
for r in recipes:
    print(f"  [{r['template']}] {r['question']} → {r['answer']}")

with open("experiments/m344_template_results.json", "w") as f:
    json.dump({"templates": len(templates), "recipes": len(recipes)}, f, indent=2)

print("\n✅ M344: Recipe templates working")
