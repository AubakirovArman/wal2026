"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M447 — Recipe Template Library

Standardized templates for common fact types.
"""
import json

templates = {
    "capital": "The capital of {country} is {capital}.",
    "population": "The population of {city} is {population}.",
    "founder": "{company} was founded by {founder} in {year}.",
}

# Validate all templates have placeholders
valid = True
for name, tmpl in templates.items():
    has_placeholder = "{" in tmpl and "}" in tmpl
    if not has_placeholder:
        valid = False
    print(f"  {name}: {tmpl}")

print(f"\nAll templates valid: {'✅' if valid else '❌'}")
assert valid

with open("experiments/m447_template_results.json", "w") as f:
    json.dump({"templates": len(templates), "valid": valid, "pass": True}, f, indent=2)

print("\n✅ M447: Recipe template library working")
