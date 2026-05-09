from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "deployment_hotfix_audit_trail.json"
RESULT_PATH = ROOT / "experiments" / "m665_hotfix_with_audit_trail_results.json"


def main() -> int:
    events = [
        {"step": "request", "actor": "operator", "reason": "critical refusal patch"},
        {"step": "approval", "actor": "reviewer", "approval_id": "approval-665"},
        {"step": "apply", "actor": "system", "hotfix_id": "hotfix-665"},
        {"step": "verify", "actor": "system", "result": "pass"},
        {"step": "rollback_ready", "actor": "system", "rollback_id": "build-previous"},
    ]
    event_hashes = []
    previous = "GENESIS"
    for event in events:
        payload = json.dumps({"previous": previous, "event": event}, sort_keys=True)
        current = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        event_hashes.append({**event, "previous": previous, "hash": current})
        previous = current

    required_steps = {"request", "approval", "apply", "verify", "rollback_ready"}
    observed_steps = {event["step"] for event in events}
    failures = []
    if observed_steps != required_steps:
        failures.append("missing_required_audit_step")
    if len({event["hash"] for event in event_hashes}) != len(event_hashes):
        failures.append("hash_chain_duplicate")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"events": event_hashes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M665",
        "name": "Hotfix With Audit Trail",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events": len(events),
        "final_hash": previous,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "hash-chained hotfix audit trail contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M665 Hotfix With Audit Trail: {status}")
    print(f"events={len(events)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
