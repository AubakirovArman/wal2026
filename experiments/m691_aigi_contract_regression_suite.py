from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, ContractRegressionSuite, Experience, VerifiedLearningLoop  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m691_aigi_contract_regression_suite_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m691_") as tmpdir:
        system = AIGISystem(workdir=tmpdir)
        protected = {f"M691 protected fact {idx}?": f"protected answer {idx}" for idx in range(10)}
        for question, answer in protected.items():
            assert system.commit(system.compile(system.propose_memory(question=question, answer=answer)))

        suite = ContractRegressionSuite.from_contract(BehavioralContract.from_dicts(must_answer=protected))
        records.append({"name": "initial_regression_suite_passes", "passed": suite.evaluate_system(system.ask).pass_})

        good_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={"M691 new fact?": "new answer"}),
            regression_suite=suite,
        )
        good = good_loop.learn_from_experience(Experience("M691 new fact?", "old", "new answer"))
        records.append({"name": "unrelated_update_passes", "passed": good.pass_ and suite.evaluate_system(system.ask).pass_})

        protected_question = "M691 protected fact 3?"
        bad_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={protected_question: "bad answer"}),
            regression_suite=suite,
        )
        bad = bad_loop.learn_from_experience(
            Experience(protected_question, protected[protected_question], "bad answer", metadata={"allow_overwrite": True})
        )
        records.append({"name": "bad_update_fails", "passed": not bad.pass_})
        records.append({"name": "bad_update_rolled_back", "passed": bad.rolled_back})
        records.append({"name": "protected_state_restored", "passed": system.ask(protected_question).answer == protected[protected_question]})
        records.append({"name": "final_regression_suite_passes", "passed": suite.evaluate_system(system.ask).pass_})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M691",
        "name": "AIGI Contract Regression Suite",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protected_contracts": 10,
        "checks_total": len(records),
        "checks_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "docs": "docs/aigi/test_log.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m691_aigi_contract_regression_suite",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M691 — Contract Regression Suite\n\n"
            f"- Status: `{status}`\n"
            f"- Protected contracts: `10`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M691 AIGI Contract Regression Suite: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

