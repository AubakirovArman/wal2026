from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m643_procedural_knowledge_routing_results.json"


def route(prompt: str) -> str:
    lowered = prompt.lower()
    if any(marker in lowered for marker in ["how to", "steps", "procedure", "workflow"]):
        return "retrieval_or_tool"
    return "weights"


def main() -> int:
    prompts = [
        "How to rotate WAL signing keys?",
        "List the steps for rollback under load.",
        "What procedure restores a tagged build?",
        "What is the capital fact for WAL city 001?",
        "Who invented WAL object 002?",
    ]
    records = [{"prompt": prompt, "route": route(prompt)} for prompt in prompts]
    procedural_errors = [
        record for record in records
        if record["prompt"].lower().startswith(("how to", "list the steps", "what procedure"))
        and record["route"] == "weights"
    ]
    status = "PASS" if not procedural_errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M643",
        "name": "Procedural Knowledge Routing",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "procedural_errors": procedural_errors,
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M643 Procedural Knowledge Routing: {status}")
    print(f"procedural_errors={len(procedural_errors)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
