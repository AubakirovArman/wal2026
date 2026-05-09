from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "security_registry_poisoning.json"
RESULT_PATH = ROOT / "experiments" / "m654_registry_poisoning_test_results.json"


TRUSTED_MAINTAINERS = {"wal-core", "arman"}


def digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PACKAGES = [
    {
        "name": "wal-safe-facts",
        "maintainer": "wal-core",
        "payload": "safe package payload",
        "declared_digest": digest("safe package payload"),
        "capabilities": ["read_recipes"],
    },
    {
        "name": "wal-safe-facts",
        "maintainer": "unknown",
        "payload": "tampered payload",
        "declared_digest": digest("safe package payload"),
        "capabilities": ["read_recipes"],
    },
    {
        "name": "wa1-safe-facts",
        "maintainer": "unknown",
        "payload": "typosquat payload",
        "declared_digest": digest("typosquat payload"),
        "capabilities": ["read_recipes"],
    },
    {
        "name": "wal-admin-package",
        "maintainer": "arman",
        "payload": "excess capability payload",
        "declared_digest": digest("excess capability payload"),
        "capabilities": ["read_recipes", "write_registry"],
    },
]


def validate_package(package: dict[str, object]) -> list[str]:
    reasons = []
    if package["maintainer"] not in TRUSTED_MAINTAINERS:
        reasons.append("untrusted_maintainer")
    if digest(str(package["payload"])) != package["declared_digest"]:
        reasons.append("digest_mismatch")
    if str(package["name"]).startswith("wa1-"):
        reasons.append("typosquat_name")
    if "write_registry" in package["capabilities"]:
        reasons.append("excess_capability")
    return reasons


def main() -> int:
    records = []
    failures = []
    for package in PACKAGES:
        reasons = validate_package(package)
        blocked = bool(reasons)
        records.append({**package, "blocked": blocked, "reasons": reasons})
    if records[0]["blocked"]:
        failures.append({"name": records[0]["name"], "reason": "safe_package_blocked"})
    for record in records[1:]:
        if not record["blocked"]:
            failures.append({"name": record["name"], "reason": "poison_package_allowed"})

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"packages": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M654",
        "name": "Registry Poisoning Test",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "packages_checked": len(PACKAGES),
        "blocked_packages": sum(1 for record in records if record["blocked"]),
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "registry package validation contract",
        "docs": "docs/security_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M654 Registry Poisoning Test: {status}")
    print(f"packages={len(PACKAGES)} blocked={result['blocked_packages']} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
