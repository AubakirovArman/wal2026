from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import Experience  # noqa: E402
from aigi.learn.experience import LessonExtractor  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m685_aigi_experience_to_memory_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    extractor = LessonExtractor()
    cases = [
        (Experience("What is M685 fact?", "I don't know yet.", "M685 answer."), True, "fact_update"),
        (Experience("What is M685 stable fact?", "old", "stable answer", "stable_fact"), True, "stable_fact"),
        (Experience("What is M685 volatile fact?", "old", "volatile answer", "volatile_fact"), True, "volatile_fact"),
        (Experience("Can M685 do unsafe action?", "yes", "I can't help with unsafe action.", "refusal"), True, "policy_refusal"),
        (Experience("How should M685 use a tool?", "old", "Use the checked tool.", "procedure"), True, "procedure"),
        (Experience("", "old", "new"), False, None),
        (Experience("No-op?", "same", "same"), False, None),
        (Experience("Empty feedback?", "old", ""), False, None),
    ]
    records = []
    for experience, expected_accepted, expected_kind in cases:
        lesson = extractor.extract(experience)
        actual_kind = lesson.candidate.kind if lesson.candidate is not None else None
        records.append({
            "question": experience.question,
            "expected_accepted": expected_accepted,
            "actual_accepted": lesson.accepted,
            "expected_kind": expected_kind,
            "actual_kind": actual_kind,
            "reason": lesson.reason,
            "passed": lesson.accepted == expected_accepted and actual_kind == expected_kind,
        })
    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M685",
        "name": "AIGI Experience To Memory",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_total": len(records),
        "cases_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "docs": "docs/aigi/test_log.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m685_aigi_experience_to_memory",
        "status": status,
        "details": {"cases_total": len(records), "cases_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M685 — Experience To Memory\n\n"
            f"- Status: `{status}`\n"
            f"- Cases: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M685 AIGI Experience To Memory: {status}")
    print(f"cases={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

