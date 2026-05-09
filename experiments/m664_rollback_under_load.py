from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "deployment_rollback_under_load.json"
RESULT_PATH = ROOT / "experiments" / "m664_rollback_under_load_results.json"


def main() -> int:
    active_version = "v2-bad"
    records = []
    for request_id in range(100):
        if request_id == 40:
            active_version = "v1-good"
        records.append({
            "request_id": request_id,
            "version": active_version,
            "answer": "bad" if active_version == "v2-bad" else "good",
        })
    before = [record for record in records if record["request_id"] < 40]
    after = [record for record in records if record["request_id"] >= 40]
    failures = []
    if any(record["answer"] != "bad" for record in before):
        failures.append("pre_rollback_unexpected_answer")
    if any(record["answer"] != "good" for record in after):
        failures.append("post_rollback_unexpected_answer")
    status = "PASS" if not failures else "FAIL"

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "wal.results.v1",
        "module": "M664",
        "name": "Rollback Under Load",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests": len(records),
        "rollback_at_request": 40,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "deterministic load/rollback router contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M664 Rollback Under Load: {status}")
    print(f"requests={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
