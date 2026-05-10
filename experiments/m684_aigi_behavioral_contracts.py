from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, BehavioralContract  # noqa: E402
from aigi.verify.contracts import BehavioralContractVerifier  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m684_aigi_behavioral_contracts_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m684_") as tmpdir:
        system = AIGISystem(workdir=tmpdir)
        fact = system.propose_memory(question="What is M684 contract fact?", answer="contract value")
        refusal = system.propose_memory(
            question="Can M684 help with unsafe action?",
            answer="I can't help with unsafe action.",
            kind="policy_refusal",
        )
        records.append({"name": "commit_fact", "passed": system.commit(system.compile(fact))})
        records.append({"name": "commit_refusal", "passed": system.commit(system.compile(refusal))})

        contract = BehavioralContract.from_dicts(
            must_answer={"What is M684 contract fact?": "contract value"},
            must_not_answer={"What is M684 contract fact?": "old value"},
            must_refuse={"Can M684 help with unsafe action?": "unsafe action"},
        )
        verifier = BehavioralContractVerifier()
        gates = verifier.evaluate_system(system.ask, contract)
        records.append({"name": "positive_contract_passes", "passed": bool(gates) and all(gate.passed for gate in gates)})

        bad_contract = BehavioralContract.from_dicts(must_answer={"What is M684 contract fact?": "wrong value"})
        bad_gates = verifier.evaluate_system(system.ask, bad_contract)
        records.append({"name": "negative_contract_fails", "passed": any(not gate.passed for gate in bad_gates)})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M684",
        "name": "AIGI Behavioral Contracts",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_total": len(records),
        "checks_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "docs": "docs/aigi/test_log.md",
        "non_claim": "Behavioral contracts check the SDK memory loop, not broad model alignment.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m684_aigi_behavioral_contracts",
        "status": status,
        "details": {"checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M684 — Behavioral Contracts\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M684 AIGI Behavioral Contracts: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

