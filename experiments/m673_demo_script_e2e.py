from __future__ import annotations

import contextlib
import io
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "product_demo_e2e_output.txt"
RESULT_PATH = ROOT / "experiments" / "m673_demo_script_e2e_results.json"


def main() -> int:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_path(str(ROOT / "wal_studio_v01" / "demo.py"), run_name="__main__")
    output = stdout.getvalue()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(output, encoding="utf-8")
    checks = [
        {"name": "demo_complete", "passed": "DEMO COMPLETE" in output},
        {"name": "bad_edit_caught", "passed": "Bad edit caught" in output},
        {"name": "rollback_pass", "passed": "rollback v1.0" in output.lower() and "CI gate: PASS" in output},
        {"name": "twelve_steps", "passed": output.count("STEP ") == 12},
    ]
    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M673",
        "name": "Demo Script E2E",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "WAL Studio demo execution gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M673 Demo Script E2E: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
