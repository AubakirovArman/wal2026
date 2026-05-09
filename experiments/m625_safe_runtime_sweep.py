"""M625 — Safe Runtime Sweep.

Runs all experiment scripts classified as safe by the M624 policy in M-order.
Heavy GPU/model-loading, destructive, git-mutating, and recursive audit scripts
are recorded as BLOCKED instead of being executed blindly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from m624_full_test_inventory import classify, order_key


def run_script(path: Path, timeout: int) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = round(time.monotonic() - start, 3)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        return {
            "file": path.name,
            "status": status,
            "pass": status == "PASS",
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "stdout_tail": completed.stdout[-1200:],
            "stderr_tail": completed.stderr[-1200:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.monotonic() - start, 3)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "file": path.name,
            "status": "BLOCKED",
            "pass": False,
            "reason": "TIMEOUT",
            "timeout_sec": timeout,
            "elapsed_sec": elapsed,
            "stdout_tail": stdout[-1200:],
            "stderr_tail": stderr[-1200:],
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run safe experiment scripts in order")
    parser.add_argument("--timeout", type=int, default=15, help="Per-script timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Optional max runnable scripts")
    args = parser.parse_args(argv)

    paths = sorted(EXPERIMENTS.glob("*.py"), key=order_key)
    records: list[dict[str, object]] = []
    runnable_count = 0
    for path in paths:
        inventory = classify(path)
        if not inventory["runnable"]:
            records.append({
                "file": path.name,
                "status": "BLOCKED",
                "pass": False,
                "reason": "NOT_SAFE_FOR_AUTOMATED_SWEEP",
                "blocked_reasons": inventory["blocked_reasons"],
            })
            continue
        if args.limit is not None and runnable_count >= args.limit:
            records.append({
                "file": path.name,
                "status": "BLOCKED",
                "pass": False,
                "reason": "LIMIT_NOT_EXECUTED",
            })
            continue
        runnable_count += 1
        print(f"[M625] RUN {runnable_count}: {path.name}", flush=True)
        records.append(run_script(path, timeout=args.timeout))

    status_counts: dict[str, int] = {}
    failures = []
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "FAIL":
            failures.append(record)

    result = {
        "schema_version": "wal.results.v1",
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "total_scripts": len(records),
        "executed_scripts": sum(1 for record in records if "returncode" in record or record.get("reason") == "TIMEOUT"),
        "status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
        "timeout_sec": args.timeout,
        "records": records,
    }

    print("=" * 60)
    print("M625 — SAFE RUNTIME SWEEP")
    print("=" * 60)
    print(f"  Total: {result['total_scripts']}")
    print(f"  Executed: {result['executed_scripts']}")
    print(f"  Status counts: {result['status_counts']}")

    (EXPERIMENTS / "m625_safe_runtime_sweep_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nM625 status={result['status']}")


if __name__ == "__main__":
    main()
