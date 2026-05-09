"""M518 — Automated Test Suite.

Runs the maintained core pytest suite. Historical M5xx scripts include slow,
GPU-bound, and documentation-only probes, so they are not a valid automated
release gate.
"""
import json
import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
run = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests"],
    capture_output=True,
    text=True,
    timeout=120,
    env=env,
)
passed = run.returncode == 0

print("=" * 60)
print("M518 — AUTOMATED TEST SUITE")
print("=" * 60)
print(f"  Command: python -m pytest -q tests")
print(f"  Return code: {run.returncode}")
print(run.stdout.strip())
if run.stderr.strip():
    print(run.stderr.strip())

with open("experiments/m518_auto_test_results.json", "w") as f:
    json.dump({
        "schema_version": "wal.results.v1",
        "suite": "core_pytest",
        "command": "python -m pytest -q tests",
        "returncode": run.returncode,
        "stdout_tail": run.stdout[-2000:],
        "stderr_tail": run.stderr[-2000:],
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
    }, f, indent=2)

print(f"\nM518: Automated test suite status={'PASS' if passed else 'FAIL'}")
