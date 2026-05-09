"""
M582 — Project Metrics

Key metrics dashboard.
"""
import json, glob

metrics = {
    "experiments": len(glob.glob("experiments/m*.py")),
    "results": len(glob.glob("experiments/*_results.json")),
    "books": len(glob.glob("book/*.md")),
    "docs": len(glob.glob("docs/**/*.md", recursive=True)),
    "badges": 10,
    "git_tags": 2,
}

print("=" * 60)
print("M582 — PROJECT METRICS")
print("=" * 60)
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open("experiments/m582_metrics_results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ M582: Metrics generated")
