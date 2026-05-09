from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m645_hard_facts_hybrid_backend_results.json"


def choose_backend(record: dict[str, str]) -> str:
    if record["type"] in {"author", "inventor"}:
        return "retrieval_first_with_edit_fallback"
    if record["type"] == "stable_exact":
        return "weights_candidate"
    return "retrieval"


def main() -> int:
    records = [
        {"id": "hard-001", "type": "author", "question": "Who wrote obscure WAL paper 001?"},
        {"id": "hard-002", "type": "inventor", "question": "Who invented WAL device 002?"},
        {"id": "hard-003", "type": "stable_exact", "question": "What is the checksum label for WAL build 003?"},
        {"id": "hard-004", "type": "temporal", "question": "Who is current maintainer for WAL branch 004?"},
    ]
    routed = [{**record, "backend": choose_backend(record)} for record in records]
    errors = [
        record for record in routed
        if record["type"] in {"author", "inventor"} and not record["backend"].startswith("retrieval_first")
    ]
    status = "SIMULATED" if not errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M645",
        "name": "Hard Facts Hybrid Backend",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": routed,
        "routing_errors": errors,
        "claim": "routing contract only; no real hybrid backend execution",
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M645 Hard Facts Hybrid Backend: {status}")
    print(f"routing_errors={len(errors)}")
    return 0 if status in {"SIMULATED", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
