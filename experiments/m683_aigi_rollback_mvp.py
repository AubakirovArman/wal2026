from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, MemoryPolicy  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m683_aigi_rollback_mvp_results.json"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    records = []
    with tempfile.TemporaryDirectory(prefix="aigi_m683_") as tmpdir:
        system = AIGISystem(
            workdir=tmpdir,
            memory_policy=MemoryPolicy(allow_weight_tier=True),
        )
        question = "What does M683 rollback test?"
        first = system.propose_memory(question=question, answer="first", kind="stable_fact")
        first_report = system.compile(first)
        first_commit = system.commit(first_report)
        recipe_path = Path(tmpdir) / "wal_recipes" / f"{first_report.artifact_id}.json"
        records.append({
            "name": "commit_first_wal_recipe",
            "passed": first_commit and recipe_path.exists() and system.ask(question).answer == "first",
        })
        rollback_first = system.rollback_last()
        records.append({
            "name": "rollback_removes_wal_recipe",
            "passed": rollback_first and not recipe_path.exists() and system.ask(question).source == "base_model_fallback",
        })

        baseline = system.propose_memory(question=question, answer="baseline", kind="fact_update")
        records.append({"name": "commit_baseline", "passed": system.commit(system.compile(baseline))})
        overwrite = system.propose_memory(
            question=question,
            answer="overwrite",
            kind="fact_update",
            metadata={"allow_overwrite": True},
        )
        records.append({"name": "commit_overwrite", "passed": system.commit(system.compile(overwrite))})
        records.append({"name": "overwrite_visible", "passed": system.ask(question).answer == "overwrite"})
        records.append({"name": "rollback_restores_baseline", "passed": system.rollback_last() and system.ask(question).answer == "baseline"})
        records.append({"name": "rollback_removes_baseline", "passed": system.rollback_last() and system.ask(question).source == "base_model_fallback"})
        records.append({"name": "empty_history_fails", "passed": not system.rollback_last()})

    failures = [record for record in records if not record["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M683",
        "name": "AIGI Rollback MVP",
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
        "event": "m683_aigi_rollback_mvp",
        "status": status,
        "details": {
            "checks_total": len(records),
            "checks_passed": len(records) - len(failures),
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M683 — Rollback MVP\n\n"
            f"- Status: `{status}`\n"
            f"- Checks: `{len(records)}`\n"
            f"- Passed: `{len(records) - len(failures)}`\n"
        )
    print(f"M683 AIGI Rollback MVP: {status}")
    print(f"checks={len(records) - len(failures)}/{len(records)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
