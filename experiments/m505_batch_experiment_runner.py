"""
M505 — Batch Experiment Runner

Runs multiple experiments in sequence and reports aggregate results.
"""
import json, glob, subprocess, os

experiments = sorted([p for p in glob.glob("experiments/m5*.py") if "runner" not in os.path.basename(p)])[:5]
results = []

print("=" * 60)
print("M505 — BATCH EXPERIMENT RUNNER")
print("=" * 60)

for path in experiments:
    name = os.path.basename(path)
    print(f"  Running {name}...")
    r = subprocess.run(["python", path], capture_output=True, text=True, timeout=30)
    success = r.returncode == 0
    results.append({"name": name, "success": success})
    print(f"    {'✅' if success else '❌'}")

passed = sum(1 for r in results if r["success"])
print(f"\nPassed: {passed}/{len(results)}")

with open("experiments/m505_batch_runner_results.json", "w") as f:
    json.dump({"ran": len(results), "passed": passed, "pass": passed == len(results)}, f, indent=2)

print("\n✅ M505: Batch runner complete")
