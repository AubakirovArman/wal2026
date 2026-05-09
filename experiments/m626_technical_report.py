from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "TECHNICAL_REPORT.md"
RESULT_PATH = ROOT / "experiments" / "m626_technical_report_results.json"


REQUIRED_SECTIONS = [
    "## Executive Summary",
    "## Project Scope",
    "## Architecture",
    "## Validation Snapshot",
    "## Status Semantics",
    "## Strengths",
    "## Limitations",
    "## Recommended Public Claims",
    "## Next Validation Protocol",
    "## Current Release Readiness",
]

REQUIRED_PHRASES = [
    "pre-alpha research framework",
    "M624",
    "M625",
    "BLOCKED",
    "UNSUPPORTED",
    "SIMULATED",
    "cross-model",
]

PROHIBITED_PHRASES = [
    "complete and production",
    "certified A+",
]


def main() -> int:
    checks: list[dict[str, object]] = []

    exists = REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8") if exists else ""
    checks.append({"name": "report_exists", "passed": exists})

    for section in REQUIRED_SECTIONS:
        checks.append(
            {
                "name": f"section:{section}",
                "passed": section in text,
            }
        )

    lower_text = text.lower()
    for phrase in REQUIRED_PHRASES:
        checks.append(
            {
                "name": f"phrase:{phrase}",
                "passed": phrase.lower() in lower_text,
            }
        )

    for phrase in PROHIBITED_PHRASES:
        checks.append(
            {
                "name": f"prohibited:{phrase}",
                "passed": phrase.lower() not in lower_text,
            }
        )

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    status = "PASS" if failed == 0 else "FAIL"

    result = {
        "schema_version": 1,
        "module": "M626",
        "name": "Technical Report Gate",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_failed": failed,
        },
        "checks": checks,
    }

    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"M626 Technical Report Gate: {status}")
    print(f"checks={passed}/{len(checks)} report={REPORT_PATH.relative_to(ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
