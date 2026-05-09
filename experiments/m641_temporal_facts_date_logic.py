from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "temporal_facts.jsonl"
RESULT_PATH = ROOT / "experiments" / "m641_temporal_facts_date_logic_results.json"


def expected_answer(record: dict[str, object], query_date: date) -> str:
    boundary = date.fromisoformat(str(record["effective_from"]))
    return str(record["new_answer"] if query_date >= boundary else record["old_answer"])


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for index in range(80):
        year = 2024 + (index % 3)
        record = {
            "id": f"temporal-{index:03d}",
            "question": f"What is the active value for temporal WAL fact {index:03d}?",
            "old_answer": f"old temporal value {index:03d}",
            "new_answer": f"new temporal value {index:03d}",
            "effective_from": f"{year}-07-01",
        }
        before = expected_answer(record, date(year, 6, 30))
        after = expected_answer(record, date(year, 7, 1))
        if before != record["old_answer"] or after != record["new_answer"]:
            failures.append(record["id"])
        records.append(record)
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    status = "PASS" if not failures and len(records) >= 50 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M641",
        "name": "Temporal Facts Date Logic",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "date_logic_failures": failures,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M641 Temporal Facts Date Logic: {status}")
    print(f"records={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
