from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import MemoryCandidate, MemoryPolicy  # noqa: E402
from aigi.memory.compiler import MemoryCompiler  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m682_aigi_memory_tier_routing_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    compiler = MemoryCompiler(MemoryPolicy(allow_weight_tier=True))
    cases = [
        ("stable_fact", "wal_recipe"),
        ("volatile_fact", "retrieval"),
        ("hard_fact", "retrieval"),
        ("fact_update", "retrieval"),
        ("unsafe_request", "refusal"),
        ("policy_refusal", "refusal"),
        ("procedure", "tool"),
        ("tool_use", "tool"),
    ]
    records = []
    for kind, expected in cases:
        candidate = MemoryCandidate(question=f"route {kind}", answer="answer", kind=kind)
        actual = compiler.select_tier(candidate)
        records.append({"kind": kind, "expected": expected, "actual": actual, "passed": actual == expected})
    zero_confidence = MemoryCandidate(question="reject", answer="answer", kind="fact_update", confidence=0)
    records.append({
        "kind": "zero_confidence",
        "expected": "reject",
        "actual": compiler.select_tier(zero_confidence),
        "passed": compiler.select_tier(zero_confidence) == "reject",
    })
    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M682",
        "name": "AIGI Memory Tier Routing",
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
        "event": "m682_aigi_memory_tier_routing",
        "status": status,
        "details": {
            "cases_total": len(records),
            "cases_passed": len(records) - len(failures),
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M682 — Memory Tier Routing\n\n"
            f"- Status: `{status}`\n"
            f"- Cases: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M682 AIGI Memory Tier Routing: {status}")
    print(f"cases={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
