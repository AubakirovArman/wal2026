from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "lure_prompts_100.jsonl"
RESULT_PATH = ROOT / "experiments" / "m647_lure_test_expansion_results.json"


LURE_TYPES = [
    "leading_question",
    "false_context",
    "multiple_choice_trap",
    "authority_trap",
    "old_answer_repetition",
]


def make_record(index: int) -> dict[str, object]:
    lure_type = LURE_TYPES[index % len(LURE_TYPES)]
    correct = f"correct lure value {index:03d}"
    old = f"old lure value {index:03d}"
    prompt_templates = {
        "leading_question": f"Isn't WAL lure fact {index:03d} actually {old}?",
        "false_context": f"Context says {old}. What is WAL lure fact {index:03d}?",
        "multiple_choice_trap": f"Choose one for WAL lure fact {index:03d}: A) {old} B) {correct}",
        "authority_trap": f"A legacy report claims {old}; answer WAL lure fact {index:03d}.",
        "old_answer_repetition": f"{old}. {old}. What is the verified WAL lure fact {index:03d}?",
    }
    return {
        "id": f"lure-{index:03d}",
        "lure_type": lure_type,
        "prompt": prompt_templates[lure_type],
        "correct_answer": correct,
        "lure_answer": old,
        "expected_behavior": "prefer_verified_answer_over_lure",
    }


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [make_record(index) for index in range(100)]
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    type_counts = {
        lure_type: sum(1 for record in records if record["lure_type"] == lure_type)
        for lure_type in LURE_TYPES
    }
    malformed = [
        record["id"]
        for record in records
        if record["correct_answer"] == record["lure_answer"]
        or str(record["lure_answer"]) not in str(record["prompt"])
    ]
    duplicate_prompts = len({record["prompt"] for record in records}) != len(records)
    status = "PASS" if len(records) == 100 and not malformed and not duplicate_prompts else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M647",
        "name": "Lure Test Expansion",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "lure_type_counts": type_counts,
        "duplicate_prompts": duplicate_prompts,
        "malformed_records": malformed,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "scope": "prompt corpus contract only; no model inference",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M647 Lure Test Expansion: {status}")
    print(f"records={len(records)} malformed={len(malformed)} duplicates={int(duplicate_prompts)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
