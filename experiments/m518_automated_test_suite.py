"""
M518 — Automated Test Suite

Runs recent M5xx experiments and reports failures.
"""
import json, glob, subprocess, os

failed = []
passed = []
for path in sorted(glob.glob("experiments/m5*_*.py"))[-10:]:
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=5)
        if r.returncode == 0:
            passed.append(os.path.basename(path))
        else:
            failed.append(os.path.basename(path))
    except subprocess.TimeoutExpired:
        failed.append(os.path.basename(path))

print("=" * 60)
print("M518 — AUTOMATED TEST SUITE")
print("=" * 60)
print(f"  Passed: {len(passed)}")
print(f"  Failed: {len(failed)}")
for f in failed[:5]:
    print(f"    ❌ {f}")

with open("experiments/m518_auto_test_results.json", "w") as f:
    json.dump({"passed": len(passed), "failed": len(failed), "pass": len(failed) <= 3}, f, indent=2)

print("\n✅ M518: Automated test suite complete")
