from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, CommitDecisionReporter, Experience, RiskLedger, VerifiedLearningLoop  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m690_aigi_risk_ledger_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m690_") as tmpdir:
        workdir = Path(tmpdir)
        system = AIGISystem(workdir=workdir / "system")
        ledger = RiskLedger(workdir / "risk.jsonl")
        reporter = CommitDecisionReporter(workdir / "decisions.jsonl")

        accepted_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={"M690 accepted?": "yes"}),
            risk_ledger=ledger,
            decision_reporter=reporter,
        )
        accepted = accepted_loop.learn_from_experience(Experience("M690 accepted?", "old", "yes"))
        records.append({"name": "accepted_entry", "passed": accepted.pass_ and accepted.committed})

        protected = "M690 protected?"
        assert system.commit(system.compile(system.propose_memory(question=protected, answer="baseline")))
        rollback_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={protected: "baseline"}),
            risk_ledger=ledger,
            decision_reporter=reporter,
        )
        rolled_back = rollback_loop.learn_from_experience(
            Experience(protected, "baseline", "bad", metadata={"allow_overwrite": True})
        )
        records.append({"name": "rolled_back_entry", "passed": not rolled_back.pass_ and rolled_back.rolled_back})

        rejected_loop = VerifiedLearningLoop(system, risk_ledger=ledger, decision_reporter=reporter)
        rejected = rejected_loop.learn_from_experience(
            Experience(protected, "baseline", "bad again", metadata={"allow_overwrite": True})
        )
        records.append({"name": "rejected_entry", "passed": not rejected.pass_ and rejected.risk_score > 0})

        summary = ledger.summary()
        decisions = reporter.load()
        records.append({"name": "ledger_entries_count", "passed": summary["entries"] == 3})
        records.append({"name": "active_debt_matches", "passed": summary["active_debt"] == accepted.risk_score})
        records.append({"name": "rolled_back_debt_matches", "passed": summary["rolled_back_debt"] == rolled_back.risk_score})
        records.append({"name": "rejected_debt_matches", "passed": summary["rejected_debt"] == rejected.risk_score})
        records.append({"name": "decision_sequence", "passed": [decision.decision for decision in decisions] == ["ACCEPTED", "ROLLED_BACK", "REJECTED"]})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M690",
        "name": "AIGI Risk Ledger",
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
        "event": "m690_aigi_risk_ledger",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M690 — Risk Ledger\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M690 AIGI Risk Ledger: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

