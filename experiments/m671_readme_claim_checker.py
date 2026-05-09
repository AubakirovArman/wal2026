from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m671_readme_claim_checker_results.json"


FILES = ["README.md", "PROJECT_SUMMARY.md", "TECHNICAL_REPORT.md", "RELEASE_NOTES_v2.md"]
FORBIDDEN = [
    re.compile(r"\bproduction[- ]ready\b", re.IGNORECASE),
    re.compile(r"certified\s+A\+", re.IGNORECASE),
    re.compile(r"complete\s+and\s+production", re.IGNORECASE),
]


def main() -> int:
    failures = []
    for rel_path in FILES:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                failures.append({"file": rel_path, "pattern": pattern.pattern})
        if rel_path == "README.md" and "pre-alpha" not in text.lower():
            failures.append({"file": rel_path, "pattern": "missing_pre_alpha"})
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M671",
        "name": "README Claim Checker",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_checked": FILES,
        "failures": failures,
        "scope": "release-facing claim sanity gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M671 README Claim Checker: {status}")
    print(f"files={len(FILES)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
