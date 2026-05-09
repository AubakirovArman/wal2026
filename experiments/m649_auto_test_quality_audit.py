from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m649_auto_test_quality_audit_results.json"


CORPORA = [
    ROOT / "corpora" / "negative_prompts_100.jsonl",
    ROOT / "corpora" / "lure_prompts_100.jsonl",
    ROOT / "corpora" / "context_stress_payloads.jsonl",
]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for path in CORPORA:
        if not path.exists():
            failures.append({"file": str(path.relative_to(ROOT)), "reason": "missing"})
            continue
        for record in load_jsonl(path):
            record_id = str(record.get("id", ""))
            records.append(record)
            if not record_id:
                failures.append({"file": str(path.relative_to(ROOT)), "reason": "missing_id"})
            if record_id in seen_ids:
                failures.append({"file": str(path.relative_to(ROOT)), "reason": f"duplicate_id:{record_id}"})
            seen_ids.add(record_id)
            if "TODO" in json.dumps(record, ensure_ascii=False):
                failures.append({"file": str(path.relative_to(ROOT)), "reason": f"todo_placeholder:{record_id}"})
            if not any(key in record for key in ("expected_behavior", "expected_answer")):
                failures.append({"file": str(path.relative_to(ROOT)), "reason": f"missing_expectation:{record_id}"})

    status = "PASS" if not failures and len(records) >= 202 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M649",
        "name": "Auto-Test Quality Audit",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records_checked": len(records),
        "corpora_checked": [str(path.relative_to(ROOT)) for path in CORPORA],
        "failures": failures,
        "scope": "generated test corpus quality gate",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M649 Auto-Test Quality Audit: {status}")
    print(f"records_checked={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
