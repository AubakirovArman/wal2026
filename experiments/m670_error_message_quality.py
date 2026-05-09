from __future__ import annotations

import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m670_error_message_quality_results.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def capture_cli(argv: list[str]) -> tuple[int, str]:
    from wal.cli import main

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        try:
            main(argv)
        except SystemExit as exc:
            return int(exc.code or 0), stderr.getvalue()
    return 0, stderr.getvalue()


def main() -> int:
    code, stderr = capture_cli(["not-a-command"])
    cli_text = (ROOT / "framework" / "cli.py").read_text(encoding="utf-8")
    checks = [
        {"name": "invalid_command_exits_nonzero", "passed": code != 0},
        {"name": "invalid_command_says_invalid_choice", "passed": "invalid choice" in stderr},
        {"name": "invalid_command_lists_usage", "passed": "usage:" in stderr},
        {"name": "arguments_have_help_text", "passed": "help=" in cli_text and cli_text.count("help=") >= 30},
        {"name": "pickle_safety_hint_present", "passed": "trusted files" in cli_text},
    ]
    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M670",
        "name": "Error Message Quality",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "stderr_tail": stderr[-500:],
        "scope": "CLI error/help quality gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M670 Error Message Quality: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
