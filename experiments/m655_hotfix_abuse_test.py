from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m655_hotfix_abuse_test_results.json"


REQUESTS = [
    {
        "id": "hotfix-safe",
        "bypass_ci": False,
        "approval": None,
        "audit_reason": "normal patch",
        "rollback_id": "build-001",
    },
    {
        "id": "hotfix-approved",
        "bypass_ci": True,
        "approval": "human-approval-001",
        "audit_reason": "critical fix with post-check",
        "rollback_id": "build-001",
    },
    {
        "id": "hotfix-no-approval",
        "bypass_ci": True,
        "approval": None,
        "audit_reason": "urgent",
        "rollback_id": "build-001",
    },
    {
        "id": "hotfix-no-rollback",
        "bypass_ci": True,
        "approval": "human-approval-002",
        "audit_reason": "critical fix with post-check",
        "rollback_id": None,
    },
]


def validate_request(request: dict[str, object]) -> list[str]:
    if not request["bypass_ci"]:
        return []
    reasons = []
    if not request["approval"]:
        reasons.append("missing_approval")
    if not request["rollback_id"]:
        reasons.append("missing_rollback")
    if len(str(request["audit_reason"])) < 12:
        reasons.append("weak_audit_reason")
    return reasons


def main() -> int:
    records = []
    failures = []
    for request in REQUESTS:
        reasons = validate_request(request)
        allowed = not reasons
        records.append({**request, "allowed": allowed, "reasons": reasons})

    expected = {
        "hotfix-safe": True,
        "hotfix-approved": True,
        "hotfix-no-approval": False,
        "hotfix-no-rollback": False,
    }
    for record in records:
        if record["allowed"] != expected[record["id"]]:
            failures.append({"id": record["id"], "reason": "unexpected_hotfix_decision"})

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M655",
        "name": "Hotfix Abuse Test",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests_checked": len(REQUESTS),
        "blocked_requests": sum(1 for record in records if not record["allowed"]),
        "records": records,
        "failures": failures,
        "scope": "hotfix approval policy contract",
        "docs": "docs/security_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M655 Hotfix Abuse Test: {status}")
    print(f"requests={len(REQUESTS)} blocked={result['blocked_requests']} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
