from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, MemoryPolicy  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m680_aigi_100_fact_learning_loop_results.json"
LOG_PATH = ROOT / "logs" / "aigi" / "m680_100_fact_learning_loop.jsonl"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    start = time.monotonic()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    facts = [
        (
            f"What is AIGI synthetic fact {idx:03d}?",
            f"AIGI synthetic answer {idx:03d}.",
            "stable_fact" if idx % 2 == 0 else "fact_update",
        )
        for idx in range(100)
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m680_") as tmpdir:
        system = AIGISystem(
            workdir=tmpdir,
            memory_policy=MemoryPolicy(allow_weight_tier=True),
        )
        for question, answer, kind in facts:
            before = system.ask(question)
            candidate = system.propose_memory(question=question, answer=answer, kind=kind, source="m680")
            report = system.compile(candidate)
            committed = system.commit(report)
            after = system.ask(question)
            record = {
                "question": question,
                "kind": kind,
                "before_source": before.source,
                "tier": report.tier,
                "compile_pass": report.pass_,
                "committed": committed,
                "after_correct": after.answer == answer,
                "after_source": after.source,
            }
            records.append(record)
            append_jsonl(LOG_PATH, record)

    failures = [
        record for record in records
        if record["before_source"] != "base_model_fallback"
        or not record["compile_pass"]
        or not record["committed"]
        or not record["after_correct"]
    ]
    tier_counts: dict[str, int] = {}
    for record in records:
        tier_counts[record["tier"]] = tier_counts.get(record["tier"], 0) + 1
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M680",
        "name": "AIGI 100 Fact Learning Loop",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "facts_total": len(records),
        "facts_passed": len(records) - len(failures),
        "tier_counts": dict(sorted(tier_counts.items())),
        "elapsed_sec": round(time.monotonic() - start, 3),
        "failures": failures[:10],
        "log": str(LOG_PATH.relative_to(ROOT)),
        "docs": "docs/aigi/test_log.md",
        "non_claim": "M680 verifies SDK memory accumulation through recipe/retrieval layers, not real semantic weight editing.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m680_aigi_100_fact_learning_loop",
        "status": status,
        "details": {
            "facts_total": len(records),
            "facts_passed": len(records) - len(failures),
            "tier_counts": dict(sorted(tier_counts.items())),
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M680 — 100 Fact Learning Loop\n\n"
            f"- Status: `{status}`\n"
            f"- Facts: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
            f"- Tier counts: `{dict(sorted(tier_counts.items()))}`\n"
        )
    print(f"M680 AIGI 100 Fact Learning Loop: {status}")
    print(f"facts={len(records) - len(failures)}/{len(records)} tiers={dict(sorted(tier_counts.items()))}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
