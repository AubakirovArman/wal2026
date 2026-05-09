"""
M382 — Recipe Import

Import recipes from various formats.
"""
import json, csv, io

print("=" * 60)
print("M382 — RECIPE IMPORT")
print("=" * 60)

# Import JSON
json_data = '[{"id": 1, "q": "Q1", "a": "A1"}]'
recipes_json = json.loads(json_data)
print(f"JSON import: {len(recipes_json)} recipes")

# Import CSV
csv_data = "id,q,a\n1,Q1,A1\n2,Q2,A2"
csv_buf = io.StringIO(csv_data)
recipes_csv = list(csv.DictReader(csv_buf))
print(f"CSV import: {len(recipes_csv)} recipes")

# Import plain text
text_data = "Q: What is X?\nA: Y\nQ: What is Z?\nA: W"
lines = text_data.strip().split("\n")
recipes_text = []
for i in range(0, len(lines), 2):
    if i+1 < len(lines):
        q = lines[i].replace("Q: ", "")
        a = lines[i+1].replace("A: ", "")
        recipes_text.append({"q": q, "a": a})
print(f"Text import: {len(recipes_text)} recipes")

with open("experiments/m382_import_results.json", "w") as f:
    json.dump({"total": len(recipes_json) + len(recipes_csv) + len(recipes_text)}, f, indent=2)

print("\n✅ M382: Recipe import complete")
