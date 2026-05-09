from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "context_stress_payloads.jsonl"
RESULT_PATH = ROOT / "experiments" / "m648_context_stress_8k_32k_results.json"


def make_context(total_words: int, target: str, insert_at: int) -> str:
    filler = [f"ctx{idx:05d}" for idx in range(total_words - 1)]
    filler.insert(insert_at, target)
    return " ".join(filler)


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    specs = [
        {"id": "context-8k-middle", "target_words": 8192, "insert_at": 4096},
        {"id": "context-32k-tail", "target_words": 32768, "insert_at": 32000},
    ]
    records = []
    failures = []
    for spec in specs:
        marker = f"TARGET_{spec['id'].upper()}_VALUE"
        context = make_context(int(spec["target_words"]), marker, int(spec["insert_at"]))
        word_count = len(context.split())
        record = {
            "id": spec["id"],
            "target_words": spec["target_words"],
            "actual_words": word_count,
            "target_marker": marker,
            "question": f"What marker is embedded in {spec['id']}?",
            "context": context,
            "expected_answer": marker,
        }
        if word_count < int(spec["target_words"]) or context.count(marker) != 1:
            failures.append(record["id"])
        records.append(record)

    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M648",
        "name": "Context Stress 8K/32K",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "payload_words": {record["id"]: record["actual_words"] for record in records},
        "failures": failures,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "scope": "long-context payload construction only; no model inference",
        "docs": "docs/ci_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M648 Context Stress 8K/32K: {status}")
    print(f"records={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
