from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "corpora" / "behavioral_checksum_fixtures.json"
RESULT_PATH = ROOT / "experiments" / "m651_behavioral_checksum_drift_results.json"


BASE_BEHAVIOR = [
    {"question": "What is WAL checksum fact 001?", "answer": "alpha"},
    {"question": "What is WAL checksum fact 002?", "answer": "beta"},
    {"question": "What is WAL checksum fact 003?", "answer": "gamma"},
]

SAME_BEHAVIOR_REORDERED = [
    {"answer": "gamma", "question": "What is WAL checksum fact 003?"},
    {"answer": "alpha", "question": "What is WAL checksum fact 001?"},
    {"answer": "beta", "question": "What is WAL checksum fact 002?"},
]

CHANGED_BEHAVIOR = [
    {"question": "What is WAL checksum fact 001?", "answer": "alpha"},
    {"question": "What is WAL checksum fact 002?", "answer": "changed-beta"},
    {"question": "What is WAL checksum fact 003?", "answer": "gamma"},
]


def normalize(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        [{"question": record["question"].strip(), "answer": record["answer"].strip()} for record in records],
        key=lambda record: record["question"],
    )


def checksum(records: list[dict[str, str]]) -> str:
    payload = json.dumps(normalize(records), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    base = checksum(BASE_BEHAVIOR)
    same = checksum(SAME_BEHAVIOR_REORDERED)
    changed = checksum(CHANGED_BEHAVIOR)
    failures = []
    if base != same:
        failures.append("same_behavior_checksum_changed")
    if base == changed:
        failures.append("changed_behavior_checksum_not_detected")

    fixture = {
        "base": BASE_BEHAVIOR,
        "same_behavior_reordered": SAME_BEHAVIOR_REORDERED,
        "changed_behavior": CHANGED_BEHAVIOR,
        "checksums": {
            "base": base,
            "same_behavior_reordered": same,
            "changed_behavior": changed,
        },
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M651",
        "name": "Behavioral Checksum Drift",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_checksum": base,
        "same_behavior_checksum": same,
        "changed_behavior_checksum": changed,
        "failures": failures,
        "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
        "scope": "checksum contract over behavior fixtures",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M651 Behavioral Checksum Drift: {status}")
    print(f"failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
