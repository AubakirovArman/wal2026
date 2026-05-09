"""M623 — Core Release Gate.

Runs the maintained core pytest suite as the release gate for packaged WAL APIs.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

run = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    timeout=120,
    env=env,
)

result = {
    "schema_version": "wal.results.v1",
    "experiment": "M623",
    "gate": "core_pytest",
    "command": "python -m pytest -q tests",
    "returncode": run.returncode,
    "stdout_tail": run.stdout[-2000:],
    "stderr_tail": run.stderr[-2000:],
    "status": "PASS" if run.returncode == 0 else "FAIL",
    "pass": run.returncode == 0,
}

print("=" * 60)
print("M623 — CORE RELEASE GATE")
print("=" * 60)
print(f"  Return code: {run.returncode}")
print(run.stdout.strip())
if run.stderr.strip():
    print(run.stderr.strip())

(ROOT / "experiments/m623_core_release_gate_results.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
)
print(f"\nM623 status={result['status']}")
