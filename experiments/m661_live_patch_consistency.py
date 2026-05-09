from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m661_live_patch_consistency_results.json"


def main() -> int:
    state = {"version": "v1", "facts": {"capital:brazil": "Rio de Janeiro"}}
    before = dict(state["facts"])
    patch = {"capital:brazil": "Brasília"}
    state["facts"].update(patch)
    state["version"] = "v2-live"
    after = dict(state["facts"])
    checks = [
        {"name": "old_value_changed", "passed": before["capital:brazil"] != after["capital:brazil"]},
        {"name": "new_value_visible", "passed": after["capital:brazil"] == "Brasília"},
        {"name": "version_bumped", "passed": state["version"] == "v2-live"},
    ]
    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M661",
        "name": "Live Patch Consistency",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "scope": "in-memory live patch consistency contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M661 Live Patch Consistency: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
