"""
M313 — Recipe Import/Export

Support JSON and CSV formats for recipe exchange.
"""
import json, csv, io

print("=" * 60)
print("M313 — RECIPE IMPORT/EXPORT")
print("=" * 60)

# Original recipes
recipes = [
    {"id": 1, "question": "What is the capital of France?", "answer": "Paris"},
    {"id": 2, "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": 3, "question": "What is the capital of Brazil?", "answer": "Brasília"},
]

# Export to JSON
json_export = json.dumps(recipes, indent=2)
print(f"\nJSON export: {len(json_export)} bytes")
print(json_export[:200] + "...")

# Export to CSV
csv_buffer = io.StringIO()
writer = csv.DictWriter(csv_buffer, fieldnames=["id", "question", "answer"])
writer.writeheader()
writer.writerows(recipes)
csv_export = csv_buffer.getvalue()
print(f"\nCSV export: {len(csv_export)} bytes")
print(csv_export[:200] + "...")

# Import from JSON
imported_json = json.loads(json_export)
print(f"\nJSON import: {len(imported_json)} recipes")

# Import from CSV
csv_buffer = io.StringIO(csv_export)
reader = csv.DictReader(csv_buffer)
imported_csv = [{"id": int(r["id"]), "question": r["question"], "answer": r["answer"]} for r in reader]
print(f"CSV import: {len(imported_csv)} recipes")

# Verify
assert imported_json == recipes, "JSON import mismatch"
assert imported_csv == recipes, "CSV import mismatch"

print("\nFormat comparison:")
print(f"  JSON: {len(json_export)} bytes")
print(f"  CSV:  {len(csv_export)} bytes")
print(f"  CSV is {len(json_export) / len(csv_export):.1f}× more compact")

results = {
    "recipes": len(recipes),
    "json_size": len(json_export),
    "csv_size": len(csv_export),
    "json_import_ok": imported_json == recipes,
    "csv_import_ok": imported_csv == recipes,
}

with open("experiments/m313_import_export_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M313: Recipe import/export working")
