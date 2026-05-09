from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAYBOOK_PATH = ROOT / "docs" / "demo_playbook.md"
RESULT_PATH = ROOT / "experiments" / "m627_polished_demo_playbook_results.json"


REQUIRED_STEPS = [
    "Init workspace",
    "Add good recipe",
    "Build artifact",
    "Run behavior tests",
    "Tag passing build",
    "Add bad edit",
    "CI catches failure",
    "Blame/bisect",
    "Rollback and notes",
]

REQUIRED_COMMANDS = [
    "PYTHONPATH=src python -m pytest -q tests",
    "PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid",
    "python wal_studio_v01/demo.py",
    "python experiments/m624_full_test_inventory.py",
    "python experiments/m625_safe_runtime_sweep.py",
]

REQUIRED_PHRASES = [
    "pre-alpha",
    "simulated model behavior",
    "BLOCKED",
    "UNSUPPORTED",
    "cross-model scientific validation",
]


def main() -> int:
    checks: list[dict[str, object]] = []

    exists = PLAYBOOK_PATH.exists()
    text = PLAYBOOK_PATH.read_text(encoding="utf-8") if exists else ""
    checks.append({"name": "playbook_exists", "passed": exists})

    for step in REQUIRED_STEPS:
        checks.append({"name": f"step:{step}", "passed": step in text})

    for command in REQUIRED_COMMANDS:
        checks.append({"name": f"command:{command}", "passed": command in text})

    lower_text = text.lower()
    for phrase in REQUIRED_PHRASES:
        checks.append(
            {
                "name": f"phrase:{phrase}",
                "passed": phrase.lower() in lower_text,
            }
        )

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    status = "PASS" if failed == 0 else "FAIL"

    result = {
        "schema_version": 1,
        "module": "M627",
        "name": "Polished Demo Playbook Gate",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "playbook": str(PLAYBOOK_PATH.relative_to(ROOT)),
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

    print(f"M627 Polished Demo Playbook Gate: {status}")
    print(f"checks={passed}/{len(checks)} playbook={PLAYBOOK_PATH.relative_to(ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
