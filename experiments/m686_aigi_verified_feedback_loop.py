from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, Experience, VerifiedLearningLoop  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m686_aigi_verified_feedback_loop_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m686_") as tmpdir:
        system = AIGISystem(workdir=tmpdir)
        for idx in range(25):
            question = f"What did verified feedback teach {idx:02d}?"
            answer = f"verified feedback answer {idx:02d}"
            loop = VerifiedLearningLoop(
                system,
                contract=BehavioralContract.from_dicts(
                    must_answer={question: answer},
                    must_not_answer={question: "I don't know yet."},
                ),
            )
            result = loop.learn_from_experience(
                Experience(question=question, observed_answer="I don't know yet.", feedback=answer)
            )
            records.append({
                "question": question,
                "loop_status": result.status,
                "committed": result.committed,
                "answer_correct": system.ask(question).answer == answer,
                "passed": result.pass_ and result.committed and system.ask(question).answer == answer,
            })
    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M686",
        "name": "AIGI Verified Feedback Loop",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episodes_total": len(records),
        "episodes_passed": len(records) - len(failures),
        "failures": failures,
        "docs": "docs/aigi/test_log.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m686_aigi_verified_feedback_loop",
        "status": status,
        "details": {"episodes_total": len(records), "episodes_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M686 — Verified Feedback Loop\n\n"
            f"- Status: `{status}`\n"
            f"- Episodes: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M686 AIGI Verified Feedback Loop: {status}")
    print(f"episodes={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

