from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m676_public_repo_hardening_results.json"


def contains(path: str, text: str) -> bool:
    return text in (ROOT / path).read_text(encoding="utf-8")


def missing(path: str) -> bool:
    return not (ROOT / path).exists()


def main() -> int:
    checks = [
        {
            "name": "pyproject_build_backend",
            "passed": contains("pyproject.toml", "setuptools.build_meta"),
        },
        {
            "name": "distribution_name_unique",
            "passed": contains("setup.py", 'name="wal-studio"'),
        },
        {
            "name": "cli_namespaces_present",
            "passed": contains("src/wal/cli.py", "wal core")
            and contains("src/wal/cli.py", "wal studio"),
        },
        {
            "name": "ci_release_gates_present",
            "passed": contains(".github/workflows/ci.yml", "m624_full_test_inventory.py")
            and contains(".github/workflows/ci.yml", "m625_safe_runtime_sweep.py")
            and contains(".github/workflows/ci.yml", "m630_public_claim_checker.py")
            and contains(".github/workflows/ci.yml", "m631_docs_command_smoke.py"),
        },
        {
            "name": "security_fake_email_removed",
            "passed": "security@wal-project.org" not in (ROOT / "SECURITY.md").read_text(encoding="utf-8"),
        },
        {
            "name": "historical_badges_archived",
            "passed": missing("BADGES.md")
            and missing("BADGES_FINAL.md")
            and not missing("archive/generated_history/BADGES.md")
            and not missing("archive/generated_history/BADGES_FINAL.md"),
        },
        {
            "name": "validation_status_doc_present",
            "passed": not missing("docs/VALIDATION_STATUS.md"),
        },
        {
            "name": "quickstart_example_present",
            "passed": not missing("examples/quickstart/README.md")
            and not missing("examples/quickstart/facts.json"),
        },
        {
            "name": "pages_are_product_like",
            "passed": contains("site/index.html", "30-second demo")
            and contains("site/index.html", "What is validated?")
            and contains("site/index.html", "What is not validated?"),
        },
    ]
    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M676",
        "name": "Public Repo Hardening",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "scope": "repository public-readiness hygiene gate",
        "docs": "docs/VALIDATION_STATUS.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M676 Public Repo Hardening: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
