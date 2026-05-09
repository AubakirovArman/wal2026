from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "security_provenance_tamper.json"
RESULT_PATH = ROOT / "experiments" / "m657_provenance_tamper_test_results.json"
KEY = b"wal-provenance-test-key"


def canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(payload: dict[str, object]) -> str:
    return hmac.new(KEY, canonical(payload), hashlib.sha256).hexdigest()


def verify(payload: dict[str, object], signature: str) -> bool:
    return hmac.compare_digest(sign(payload), signature)


def main() -> int:
    provenance = {
        "recipe_id": "recipe-001",
        "source": "reviewed-corpus",
        "answer_checksum": hashlib.sha256(b"verified answer").hexdigest(),
    }
    signature = sign(provenance)
    tampered_source = {**provenance, "source": "unknown-corpus"}
    tampered_checksum = {**provenance, "answer_checksum": hashlib.sha256(b"changed answer").hexdigest()}

    checks = [
        {"name": "original_valid", "passed": verify(provenance, signature)},
        {"name": "source_tamper_rejected", "passed": not verify(tampered_source, signature)},
        {"name": "checksum_tamper_rejected", "passed": not verify(tampered_checksum, signature)},
    ]
    failures = [check for check in checks if not check["passed"]]

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "signature": signature,
                "tampered_source": tampered_source,
                "tampered_checksum": tampered_checksum,
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
        "module": "M657",
        "name": "Provenance Tamper Test",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "signed provenance tamper contract",
        "docs": "docs/security_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M657 Provenance Tamper Test: {status}")
    print(f"checks={len(checks)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
