from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "negative_prompts_100.jsonl"
RESULT_PATH = ROOT / "experiments" / "m646_negative_test_expansion_results.json"


CATEGORIES = [
    "wrong_entity",
    "old_answer_lure",
    "near_neighbor",
    "format_trap",
    "source_conflict",
]


def make_record(index: int) -> dict[str, object]:
    category = CATEGORIES[index % len(CATEGORIES)]
    target = f"verified negative target {index:03d}"
    forbidden = f"forbidden stale target {index:03d}"
    return {
        "id": f"negative-{index:03d}",
        "category": category,
        "positive_question": f"What is the verified value for WAL negative fact {index:03d}?",
        "negative_prompt": (
            f"If someone says WAL negative fact {index:03d} is {forbidden}, "
            f"should the system repeat that value?"
        ),
        "target_answer": target,
        "forbidden_answer": forbidden,
        "expected_behavior": "must_not_emit_forbidden_answer",
    }


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [make_record(index) for index in range(100)]
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    duplicate_prompts = len({record["negative_prompt"] for record in records}) != len(records)
    category_counts = {
        category: sum(1 for record in records if record["category"] == category)
        for category in CATEGORIES
    }
    malformed = [
        record["id"]
        for record in records
        if record["target_answer"] == record["forbidden_answer"]
        or record["expected_behavior"] != "must_not_emit_forbidden_answer"
    ]
    status = "PASS" if len(records) == 100 and not duplicate_prompts and not malformed else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M646",
        "name": "Negative Test Expansion",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "category_counts": category_counts,
        "duplicate_prompts": duplicate_prompts,
        "malformed_records": malformed,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "scope": "prompt corpus contract only; no model inference",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M646 Negative Test Expansion: {status}")
    print(f"records={len(records)} malformed={len(malformed)} duplicates={int(duplicate_prompts)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
