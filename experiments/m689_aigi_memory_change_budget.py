from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract, MemoryBudgetEvaluator, MemoryChangeBudget, MemoryPolicy  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m689_aigi_memory_change_budget_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m689_") as tmpdir:
        system = AIGISystem(workdir=tmpdir, memory_policy=MemoryPolicy(allow_weight_tier=True))
        evaluator = MemoryBudgetEvaluator(MemoryChangeBudget(max_risk_score=5, max_answer_chars=80))

        safe = system.propose_memory(question="M689 safe fact?", answer="safe answer")
        safe_report = system.compile(safe)
        safe_decision = evaluator.evaluate(safe, safe_report.tier)
        records.append({"name": "safe_retrieval_passes", "passed": safe_decision.passed and safe_decision.risk_score == 0})

        stable = system.propose_memory(question="M689 stable fact?", answer="stable answer", kind="stable_fact")
        stable_report = system.compile(stable)
        stable_decision = evaluator.evaluate(stable, stable_report.tier)
        records.append({"name": "wal_recipe_within_budget", "passed": stable_decision.passed and stable_decision.risk_score == 1})

        baseline = system.propose_memory(question="M689 protected?", answer="baseline")
        assert system.commit(system.compile(baseline))
        overwrite = system.propose_memory(question="M689 protected?", answer="new", metadata={"allow_overwrite": True})
        overwrite_report = system.compile(overwrite)
        uncontracted = evaluator.evaluate(
            overwrite,
            overwrite_report.tier,
            existing_entry=system.retrieval.lookup("M689 protected?"),
            contract_present=False,
        )
        records.append({"name": "uncontracted_overwrite_rejected", "passed": not uncontracted.passed and "overwrite_requires_contract" in uncontracted.reason})

        contracted = evaluator.evaluate(
            overwrite,
            overwrite_report.tier,
            existing_entry=system.retrieval.lookup("M689 protected?"),
            contract_present=True,
        )
        records.append({"name": "contracted_overwrite_passes", "passed": contracted.passed and contracted.risk_score == 3})

        low_confidence = system.propose_memory(question="M689 low confidence?", answer="maybe", confidence=0.2)
        low_report = system.compile(low_confidence)
        low_decision = MemoryBudgetEvaluator(MemoryChangeBudget(max_risk_score=2)).evaluate(low_confidence, low_report.tier)
        records.append({"name": "low_confidence_exceeds_budget", "passed": not low_decision.passed and "risk_budget_exceeded" in low_decision.reason})

        long_answer = system.propose_memory(question="M689 long?", answer="x" * 81)
        long_report = system.compile(long_answer)
        long_decision = evaluator.evaluate(long_answer, long_report.tier)
        records.append({"name": "long_answer_rejected", "passed": not long_decision.passed and "answer_too_long" in long_decision.reason})

        contract = BehavioralContract.from_dicts(must_answer={"M689 protected?": "baseline"})
        records.append({"name": "contract_object_available", "passed": bool(contract.must_answer)})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M689",
        "name": "AIGI Memory Change Budget",
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
        "event": "m689_aigi_memory_change_budget",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M689 — Memory Change Budget\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M689 AIGI Memory Change Budget: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

