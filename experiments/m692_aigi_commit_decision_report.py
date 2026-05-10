from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, CommitDecisionReporter, Experience, RiskLedger, VerifiedLearningLoop  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m692_aigi_commit_decision_report_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m692_") as tmpdir:
        workdir = Path(tmpdir)
        system = AIGISystem(workdir=workdir / "system")
        reporter = CommitDecisionReporter(workdir / "decisions.jsonl")
        ledger = RiskLedger(workdir / "risk.jsonl")

        accepted_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={"M692 accepted?": "yes"}),
            risk_ledger=ledger,
            decision_reporter=reporter,
        )
        accepted = accepted_loop.learn_from_experience(Experience("M692 accepted?", "old", "yes"))
        records.append({"name": "accepted_report_present", "passed": accepted.decision_report is not None and accepted.decision_report.decision == "ACCEPTED"})

        protected = "M692 protected?"
        assert system.commit(system.compile(system.propose_memory(question=protected, answer="baseline")))
        rejected_loop = VerifiedLearningLoop(system, risk_ledger=ledger, decision_reporter=reporter)
        rejected = rejected_loop.learn_from_experience(
            Experience(protected, "baseline", "bad", metadata={"allow_overwrite": True})
        )
        records.append({"name": "rejected_report_present", "passed": rejected.decision_report is not None and rejected.decision_report.decision == "REJECTED"})

        rollback_loop = VerifiedLearningLoop(
            system,
            contract=BehavioralContract.from_dicts(must_answer={protected: "baseline"}),
            risk_ledger=ledger,
            decision_reporter=reporter,
        )
        rolled_back = rollback_loop.learn_from_experience(
            Experience(protected, "baseline", "bad", metadata={"allow_overwrite": True})
        )
        records.append({"name": "rolled_back_report_present", "passed": rolled_back.decision_report is not None and rolled_back.decision_report.decision == "ROLLED_BACK"})

        reports = reporter.load()
        records.append({"name": "three_reports_written", "passed": len(reports) == 3})
        records.append({"name": "all_reports_have_gate_counts", "passed": all(report.compile_gate_total >= report.compile_gate_passed >= 0 for report in reports)})
        records.append({"name": "all_reports_have_budget_status", "passed": all(report.budget_status in {"PASS", "FAIL"} for report in reports)})
        records.append({"name": "risk_scores_recorded", "passed": [report.risk_score for report in reports] == [accepted.risk_score, rejected.risk_score, rolled_back.risk_score]})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M692",
        "name": "AIGI Commit Decision Report",
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
        "event": "m692_aigi_commit_decision_report",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M692 — Commit Decision Report\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M692 AIGI Commit Decision Report: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

