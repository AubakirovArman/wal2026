from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "security_signed_package_verification.json"
RESULT_PATH = ROOT / "experiments" / "m658_signed_package_verification_results.json"
KEY = b"wal-package-test-key"


def canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(payload: dict[str, object]) -> str:
    return hmac.new(KEY, canonical(payload), hashlib.sha256).hexdigest()


def verify(payload: dict[str, object], signature: str) -> bool:
    return hmac.compare_digest(sign(payload), signature)


def main() -> int:
    package = {
        "name": "wal-security-fixture",
        "version": "0.1.0",
        "recipe_digest": hashlib.sha256(b"recipe payload").hexdigest(),
        "capabilities": ["read_recipes"],
    }
    signature = sign(package)
    tampered_digest = {**package, "recipe_digest": hashlib.sha256(b"tampered payload").hexdigest()}
    tampered_capabilities = {**package, "capabilities": ["read_recipes", "write_registry"]}

    checks = [
        {"name": "original_package_valid", "passed": verify(package, signature)},
        {"name": "digest_tamper_rejected", "passed": not verify(tampered_digest, signature)},
        {"name": "capability_tamper_rejected", "passed": not verify(tampered_capabilities, signature)},
    ]
    failures = [check for check in checks if not check["passed"]]

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(
            {
                "package": package,
                "signature": signature,
                "tampered_digest": tampered_digest,
                "tampered_capabilities": tampered_capabilities,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M658",
        "name": "Signed Package Verification",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "signed package verification contract",
        "docs": "docs/security_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M658 Signed Package Verification: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
