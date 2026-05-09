from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m672_docs_to_code_consistency_results.json"


COMMAND_RE = re.compile(r"^python (?P<path>experiments/[A-Za-z0-9_]+\.py|wal_studio_v01/demo\.py)$")


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = []
    missing = []
    for line in readme.splitlines():
        match = COMMAND_RE.match(line.strip())
        if not match:
            continue
        rel_path = match.group("path")
        commands.append(rel_path)
        if not (ROOT / rel_path).exists():
            missing.append(rel_path)
    required = [
        "experiments/m631_docs_command_smoke.py",
        "experiments/m668_log_volume_storage_growth.py",
        "wal_studio_v01/demo.py",
    ]
    failures = [{"path": path, "reason": "missing"} for path in missing]
    for rel_path in required:
        if rel_path not in commands:
            failures.append({"path": rel_path, "reason": "not_documented"})
    status = "PASS" if not failures and len(commands) >= 35 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M672",
        "name": "Docs-to-Code Consistency",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commands_checked": len(commands),
        "failures": failures,
        "scope": "README command target existence gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M672 Docs-to-Code Consistency: {status}")
    print(f"commands={len(commands)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
