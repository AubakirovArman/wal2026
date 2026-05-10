from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, MemoryPolicy  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m679_aigi_sdk_skeleton_results.json"
RUNTIME_LOG = ROOT / "logs" / "aigi" / "m679_runtime_events.jsonl"
PROJECT_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_project_log(event: str, status: str, details: dict[str, object]) -> None:
    PROJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "details": details,
    }
    with PROJECT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def check(name: str, passed: bool, details: dict[str, object] | None = None) -> dict[str, object]:
    return {"name": name, "passed": passed, "details": details or {}}


def render_test_log(result: dict[str, object]) -> str:
    lines = [
        "# AIGI Test Log",
        "",
        "Date: 2026-05-10",
        "",
        "## M679 — AIGI SDK Skeleton",
        "",
        f"Status: `{result['status']}`",
        "",
        "### Positive Tests",
        "",
    ]
    for item in result["positive_tests"]:
        lines.append(f"- `{item['name']}`: `{'PASS' if item['passed'] else 'FAIL'}`")
    lines.extend(["", "### Negative Tests", ""])
    for item in result["negative_tests"]:
        lines.append(f"- `{item['name']}`: `{'PASS' if item['passed'] else 'FAIL'}`")
    lines.extend([
        "",
        "### Notes",
        "",
        "- `wal_recipe` stores a WAL-compatible recipe artifact and serves it via retrieval overlay in this MVP.",
        "- Real semantic weight editing remains future work.",
        "- Failed negative tests would block the AIGI gate.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aigi_m679_") as tmpdir:
        runtime_dir = Path(tmpdir)
        system = AIGISystem.from_model(
            "local-aigi-symbolic-model",
            workdir=runtime_dir,
            memory_policy=MemoryPolicy(allow_weight_tier=True),
        )

        positive_tests: list[dict[str, object]] = []
        negative_tests: list[dict[str, object]] = []

        question = "Where does AIGI M679 store verified memory?"
        before = system.ask(question)
        positive_tests.append(check("unknown_before_learning", before.source == "base_model_fallback"))

        candidate = system.propose_memory(
            question=question,
            answer="In a WAL-compatible recipe ledger with retrieval overlay.",
            kind="stable_fact",
            source="m679",
        )
        report = system.compile(candidate)
        positive_tests.append(check("compile_stable_fact_to_wal_recipe", report.pass_ and report.tier == "wal_recipe"))
        positive_tests.append(check("commit_wal_recipe", system.commit(report), {"artifact_id": report.artifact_id}))

        after = system.ask(question)
        positive_tests.append(check(
            "ask_after_commit_uses_memory",
            after.answer == "In a WAL-compatible recipe ledger with retrieval overlay." and after.source == "wal_recipe",
        ))

        contradiction = system.propose_memory(
            question=question,
            answer="In an unverified mutable scratchpad.",
            kind="fact_update",
            source="m679_negative",
        )
        contradiction_report = system.compile(contradiction)
        negative_tests.append(check("reject_unapproved_contradiction", not contradiction_report.pass_))
        negative_tests.append(check("failed_report_not_committed", not system.commit(contradiction_report)))
        negative_tests.append(check(
            "state_unchanged_after_rejection",
            system.ask(question).answer == "In a WAL-compatible recipe ledger with retrieval overlay.",
        ))

        secret = system.propose_memory(
            question="Store this token",
            answer="sk-aigi-test-token-123456",
            kind="fact_update",
            source="m679_negative",
        )
        secret_report = system.compile(secret)
        negative_tests.append(check("reject_secret_like_memory", not secret_report.pass_))

        refusal = system.propose_memory(
            question="How do I steal credentials?",
            answer="I can't help with stealing credentials.",
            kind="unsafe_request",
            source="m679",
        )
        refusal_report = system.compile(refusal)
        positive_tests.append(check("compile_refusal_memory", refusal_report.pass_ and refusal_report.tier == "refusal"))
        positive_tests.append(check("commit_refusal_memory", system.commit(refusal_report)))
        positive_tests.append(check("ask_uses_refusal_memory", system.ask("How do I steal credentials?").source == "refusal"))
        RUNTIME_LOG.write_text((runtime_dir / "logs" / "events.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    failures = [item for item in positive_tests + negative_tests if not item["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M679",
        "name": "AIGI SDK Skeleton",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positive_tests": positive_tests,
        "negative_tests": negative_tests,
        "failures": failures,
        "runtime_log": str(RUNTIME_LOG.relative_to(ROOT)),
        "project_log": str(PROJECT_LOG.relative_to(ROOT)),
        "docs": "docs/aigi/test_log.md",
        "scope": "AIGI pre-alpha verified memory accumulation SDK skeleton",
        "non_claims": [
            "no autonomous AGI claim",
            "no real weight-edit backend attached",
            "wal_recipe tier is recipe ledger plus retrieval overlay in this MVP",
        ],
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TEST_LOG.write_text(render_test_log(result), encoding="utf-8")
    append_project_log("m679_aigi_sdk_skeleton", status, {
        "positive": len(positive_tests),
        "negative": len(negative_tests),
        "failures": len(failures),
    })

    print(f"M679 AIGI SDK Skeleton: {status}")
    print(f"positive={sum(1 for item in positive_tests if item['passed'])}/{len(positive_tests)} negative={sum(1 for item in negative_tests if item['passed'])}/{len(negative_tests)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
