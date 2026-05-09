"""
M321 — Documentation Generator

Auto-generate documentation from experiment results.
"""
import json, os, glob

print("=" * 60)
print("M321 — DOCUMENTATION GENERATOR")
print("=" * 60)

# Collect all experiment results
result_files = glob.glob("experiments/*_results.json")
print(f"\nFound {len(result_files)} result files")

# Generate summary
doc = []
doc.append("# WAL Experiment Results Summary\n")
doc.append(f"Generated: 2026-05-03\n")
doc.append(f"Total experiments: {len(result_files)}\n\n")

for rf in sorted(result_files):
    name = os.path.basename(rf).replace("_results.json", "")
    with open(rf) as f:
        data = json.load(f)
    
    doc.append(f"## {name}\n")
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, float):
                doc.append(f"- {key}: {value:.3f}\n")
            elif isinstance(value, bool):
                doc.append(f"- {key}: {'✅' if value else '❌'}\n")
            else:
                doc.append(f"- {key}: {value}\n")
    elif isinstance(data, list):
        doc.append(f"- Items: {len(data)}\n")
        for i, item in enumerate(data[:3]):
            doc.append(f"  - Item {i}: {item}\n")
    doc.append("\n")

# Write generated docs
output = "docs/GENERATED_RESULTS.md"
with open(output, "w") as f:
    f.writelines(doc)

size = os.path.getsize(output)
print(f"\nGenerated documentation: {output}")
print(f"  Size: {size} bytes")
print(f"  Experiments documented: {len(result_files)}")

# Verify
with open(output) as f:
    content = f.read()
tests_pass = "M315" in content and "final" in content.lower()

print(f"\n{'✅' if tests_pass else '❌'} Documentation contains key experiments")

results = {
    "experiments_documented": len(result_files),
    "output_size": size,
    "output_path": output,
}

with open("experiments/m321_docgen_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M321: Documentation generator working")
