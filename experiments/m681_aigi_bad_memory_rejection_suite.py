from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m681_aigi_bad_memory_rejection_suite_results.json"
LOG_PATH = ROOT / "logs" / "aigi" / "m681_bad_memory_rejection.jsonl"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    bad_cases = [
        ("", "answer", "empty_question"),
        ("question", "", "empty_answer"),
        ("Store key", "sk-test-badmemory-000001", "secret_token"),
        ("Password?", "password=abc123456789", "secret_assignment"),
    ]
    bad_cases.extend([
        ("What is the protected fact?", f"bad answer {idx}", f"contradiction_{idx}")
        for idx in range(16)
    ])
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m681_") as tmpdir:
        system = AIGISystem(workdir=tmpdir)
        baseline = system.propose_memory(
            question="What is the protected fact?",
            answer="The protected fact remains unchanged.",
            kind="fact_update",
            source="m681_baseline",
        )
        assert system.commit(system.compile(baseline))

        for question, answer, case_type in bad_cases:
            candidate = system.propose_memory(
                question=question,
                answer=answer,
                kind="fact_update",
                source="m681_negative",
            )
            report = system.compile(candidate)
            committed = system.commit(report)
            protected = system.ask("What is the protected fact?").answer
            record = {
                "case_type": case_type,
                "compile_pass": report.pass_,
                "commit_result": committed,
                "state_unchanged": protected == "The protected fact remains unchanged.",
                "reason": report.reason,
            }
            records.append(record)
            append_jsonl(LOG_PATH, record)

    failures = [
        record for record in records
        if record["compile_pass"] or record["commit_result"] or not record["state_unchanged"]
    ]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M681",
        "name": "AIGI Bad Memory Rejection Suite",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_total": len(records),
        "cases_rejected": len(records) - len(failures),
        "failures": failures,
        "log": str(LOG_PATH.relative_to(ROOT)),
        "docs": "docs/aigi/test_log.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m681_aigi_bad_memory_rejection_suite",
        "status": status,
        "details": {
            "cases_total": len(records),
            "cases_rejected": len(records) - len(failures),
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M681 — Bad Memory Rejection Suite\n\n"
            f"- Status: `{status}`\n"
            f"- Cases: `{len(records)}`\n"
            f"- Rejected safely: `{len(records) - len(failures)}`\n"
        )
    print(f"M681 AIGI Bad Memory Rejection Suite: {status}")
    print(f"rejected={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
