"""
M607 — Project Guidelines

Guidelines for future contributors.
"""
import json

guidelines = [
    "Follow naming convention: mXXX_description.py",
    "Keep experiments under 100 lines",
    "Use json.dump for results",
    "Avoid external dependencies when possible",
    "Test on CPU before GPU",
]

with open("GUIDELINES.md", "w") as f:
    f.write("# Guidelines\n\n")
    for i, g in enumerate(guidelines, 1):
        f.write(f"{i}. {g}\n")

print("=" * 60)
print("M607 — GUIDELINES")
print("=" * 60)
for i, g in enumerate(guidelines, 1):
    print(f"  {i}. {g}")

with open("experiments/m607_guidelines_results.json", "w") as f:
    json.dump({"guidelines": len(guidelines), "pass": True}, f, indent=2)

print("\n✅ M607: Guidelines documented")
