from __future__ import annotations

import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m669_cli_ux_test_results.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def capture_cli(argv: list[str]) -> tuple[int, str, str]:
    from wal.cli import main

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            main(argv)
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    help_code, help_stdout, help_stderr = capture_cli(["validate-results", "--help"])
    root_code, root_stdout, root_stderr = capture_cli([])
    checks = [
        {"name": "validate_results_help_exits_zero", "passed": help_code == 0},
        {"name": "validate_results_help_mentions_fail_on_invalid", "passed": "--fail-on-invalid" in help_stdout},
        {"name": "root_help_mentions_encode", "passed": "encode" in root_stdout},
        {"name": "root_help_nonzero_without_command", "passed": root_code == 1},
        {"name": "stderr_empty_for_help", "passed": help_stderr == ""},
    ]
    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M669",
        "name": "CLI UX Test",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "root_stderr": root_stderr[-500:],
        "scope": "CLI help and reviewer command UX gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M669 CLI UX Test: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
