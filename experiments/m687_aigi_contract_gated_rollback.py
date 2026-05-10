from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, Experience, VerifiedLearningLoop  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m687_aigi_contract_gated_rollback_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m687_") as tmpdir:
        system = AIGISystem(workdir=tmpdir)
        question = "What memory must M687 protect?"
        baseline = system.propose_memory(question=question, answer="protected baseline")
        records.append({"name": "commit_baseline", "passed": system.commit(system.compile(baseline))})

        contract = BehavioralContract.from_dicts(must_answer={question: "protected baseline"})
        loop = VerifiedLearningLoop(system, contract=contract)
        bad_update = loop.learn_from_experience(
            Experience(
                question=question,
                observed_answer="protected baseline",
                feedback="contract-breaking update",
                metadata={"allow_overwrite": True},
            )
        )
        records.append({"name": "bad_update_rejected_by_contract", "passed": not bad_update.pass_})
        records.append({"name": "bad_update_rolled_back", "passed": bad_update.rolled_back})
        records.append({"name": "baseline_restored", "passed": system.ask(question).answer == "protected baseline"})

        no_op = loop.learn_from_experience(
            Experience(question=question, observed_answer="protected baseline", feedback="protected baseline")
        )
        records.append({"name": "noop_feedback_rejected", "passed": not no_op.pass_ and no_op.reason == "feedback_matches_observed_answer"})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M687",
        "name": "AIGI Contract Gated Rollback",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_total": len(records),
        "checks_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "docs": "docs/aigi/test_log.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m687_aigi_contract_gated_rollback",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M687 — Contract-Gated Rollback\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M687 AIGI Contract Gated Rollback: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

