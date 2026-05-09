"""
M455 — Code Quality Metrics

Measures line counts, docstring coverage, and assert density.
"""
import json, glob, os

metrics = {"total_lines": 0, "docstrings": 0, "asserts": 0, "files": 0}

for path in glob.glob("experiments/m*.py")[:50]:  # sample
    with open(path) as f:
        lines = f.readlines()
    metrics["files"] += 1
    metrics["total_lines"] += len(lines)
    metrics["docstrings"] += sum(1 for l in lines if '"""' in l or "'''" in l)
    metrics["asserts"] += sum(1 for l in lines if l.strip().startswith("assert"))

print("=" * 60)
print("M455 — CODE QUALITY METRICS")
print("=" * 60)
print(f"  Files sampled: {metrics['files']}")
print(f"  Total lines: {metrics['total_lines']}")
print(f"  Docstrings: {metrics['docstrings']}")
print(f"  Asserts: {metrics['asserts']}")
print(f"  Assert density: {metrics['asserts']/max(metrics['total_lines'],1):.3f}")

with open("experiments/m455_quality_results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ M455: Code quality metrics calculated")
