"""
M575 — Project Glossary

Defines key terms used in WAL.
"""
import json

glossary = {
    "WAL": "WeightOps Abstraction Layer",
    "Recipe": "Atomic knowledge edit template",
    "CI": "Continuous Integration gate",
    "DAG": "Directed Acyclic Graph of recipes",
    "Blame": "Identify recipe causing regression",
    "Bisect": "Binary search first bad commit",
}

with open("GLOSSARY.md", "w") as f:
    f.write("# Glossary\n\n")
    for term, definition in glossary.items():
        f.write(f"**{term}**: {definition}\n\n")

print("=" * 60)
print("M575 — GLOSSARY")
print("=" * 60)
for term, definition in glossary.items():
    print(f"  {term}: {definition}")

with open("experiments/m575_glossary_results.json", "w") as f:
    json.dump({"terms": len(glossary), "pass": True}, f, indent=2)

print("\n✅ M575: Glossary generated")
