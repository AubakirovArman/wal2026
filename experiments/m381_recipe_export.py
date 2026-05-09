"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M381 — Recipe Export

Export recipes to various formats.
"""
import json, csv, io

print("=" * 60)
print("M381 — RECIPE EXPORT")
print("=" * 60)

recipes = [
    {"id": 1, "q": "Capital of France?", "a": "Paris"},
    {"id": 2, "q": "Capital of Japan?", "a": "Tokyo"},
]

# JSON
json_out = json.dumps(recipes, indent=2)
print(f"JSON export: {len(json_out)} bytes")

# CSV
csv_buf = io.StringIO()
writer = csv.DictWriter(csv_buf, fieldnames=["id", "q", "a"])
writer.writeheader()
writer.writerows(recipes)
csv_out = csv_buf.getvalue()
print(f"CSV export: {len(csv_out)} bytes")

# YAML-like
yaml_out = ""
for r in recipes:
    yaml_out += f"- id: {r['id']}\n  q: {r['q']}\n  a: {r['a']}\n"
print(f"YAML export: {len(yaml_out)} bytes")

with open("experiments/m381_export_results.json", "w") as f:
    json.dump({"formats": 3}, f, indent=2)

print("\n✅ M381: Recipe export complete")
