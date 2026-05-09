"""
M527 — Experiment Pruning

Identifies experiments that can be archived.
"""
import json, glob, os

# Find experiments older than 30 days
import time
cutoff = time.time() - 30 * 24 * 3600
old = []
for path in glob.glob("experiments/m*.py"):
    if os.path.getmtime(path) < cutoff:
        old.append(os.path.basename(path))

print("=" * 60)
print("M527 — EXPERIMENT PRUNING")
print("=" * 60)
print(f"  Experiments older than 30 days: {len(old)}")

with open("experiments/m527_pruning_results.json", "w") as f:
    json.dump({"old_experiments": len(old), "pass": True}, f, indent=2)

print("\n✅ M527: Experiment pruning analysis complete")
