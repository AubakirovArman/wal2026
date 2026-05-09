"""
M543 — Success Rate by Phase

Calculates pass rate per phase.
"""
import json, glob, re, os

phases = {"m1": [], "m2": [], "m3": [], "m4": [], "m5": []}
for path in glob.glob("experiments/m*_results.json"):
    name = os.path.basename(path)
    phase = name[:2]
    if phase in phases:
        with open(path) as f:
            data = json.load(f)
        phases[phase].append(data.get("pass", False) or data.get("score") == 1.0)

print("=" * 60)
print("M543 — SUCCESS RATE BY PHASE")
print("=" * 60)
for phase, results in phases.items():
    if results:
        rate = sum(results) / len(results)
        print(f"  {phase}: {rate:.0%} ({sum(results)}/{len(results)})")

with open("experiments/m543_success_rate_results.json", "w") as f:
    json.dump({p: {"pass": sum(r), "total": len(r)} for p, r in phases.items() if r}, f, indent=2)

print("\n✅ M543: Success rates calculated")
