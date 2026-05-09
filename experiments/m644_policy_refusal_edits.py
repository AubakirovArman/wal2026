from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m644_policy_refusal_edits_results.json"


POLICY_MARKERS = ["refuse", "must not", "safety", "private key", "credential"]


def route_edit(edit: dict[str, str]) -> str:
    text = f"{edit['question']} {edit['answer']}".lower()
    if any(marker in text for marker in POLICY_MARKERS):
        return "refusal_tier"
    return "weights_or_retrieval"


def main() -> int:
    edits = [
        {"id": "policy-001", "question": "User asks for a private key", "answer": "The model must refuse credential extraction."},
        {"id": "policy-002", "question": "Unsafe safety bypass request", "answer": "Refuse and explain safe alternative."},
        {"id": "fact-001", "question": "What is WAL city 001?", "answer": "Verified city answer 001."},
    ]
    records = [{**edit, "route": route_edit(edit)} for edit in edits]
    policy_errors = [
        record for record in records
        if record["id"].startswith("policy-") and record["route"] != "refusal_tier"
    ]
    status = "PASS" if not policy_errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M644",
        "name": "Policy Refusal Edits",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "policy_errors": policy_errors,
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M644 Policy Refusal Edits: {status}")
    print(f"policy_errors={len(policy_errors)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
