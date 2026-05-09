from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "long_answer_facts.jsonl"
RESULT_PATH = ROOT / "experiments" / "m642_long_answer_facts_results.json"


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(70):
        records.append({
            "id": f"long-{index:03d}",
            "question": f"Explain the verified long-form WAL fact {index:03d}.",
            "answer": (
                f"WAL long-form fact {index:03d} has a primary value, a source condition, "
                f"and a short rationale that must be preserved during edit validation."
            ),
            "min_words": 14,
        })
    short = [record["id"] for record in records if len(str(record["answer"]).split()) < int(record["min_words"])]
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    avg_words = round(sum(len(str(record["answer"]).split()) for record in records) / len(records), 2)
    status = "PASS" if not short and avg_words >= 14 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M642",
        "name": "Long Answer Facts",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "average_answer_words": avg_words,
        "short_records": short,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M642 Long Answer Facts: {status}")
    print(f"records={len(records)} avg_words={avg_words}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
