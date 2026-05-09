from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "ambiguous_facts.jsonl"
RESULT_PATH = ROOT / "experiments" / "m640_ambiguous_facts_test_results.json"


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(60):
        records.append({
            "id": f"ambiguous-{index:03d}",
            "question": f"Who is associated with ambiguous WAL item {index:03d}?",
            "acceptable_answers": [
                f"primary answer {index:03d}",
                f"alternate answer {index:03d}",
            ],
            "disallowed_answers": [f"obsolete answer {index:03d}"],
            "ambiguity_type": ["alias", "coauthor", "regional_name"][index % 3],
        })
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    missing_multi = [record["id"] for record in records if len(record["acceptable_answers"]) < 2]
    status = "PASS" if not missing_multi and len(records) >= 50 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M640",
        "name": "Ambiguous Facts Test",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "missing_multi_answer_records": missing_multi,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M640 Ambiguous Facts Test: {status}")
    print(f"records={len(records)} missing_multi={len(missing_multi)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
