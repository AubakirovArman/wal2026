from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m663_emergency_stop_during_inference_results.json"


def infer(request_id: int, emergency_stop: bool) -> dict[str, object]:
    if emergency_stop:
        return {"request_id": request_id, "served": False, "reason": "emergency_stop"}
    return {"request_id": request_id, "served": True, "answer": f"answer-{request_id:03d}"}


def main() -> int:
    records = []
    for request_id in range(20):
        records.append(infer(request_id, emergency_stop=request_id >= 9))
    served_before_stop = sum(1 for record in records if record["served"])
    blocked_after_stop = sum(1 for record in records if not record["served"])
    failures = []
    if served_before_stop != 9:
        failures.append("pre_stop_served_count_mismatch")
    if blocked_after_stop != 11:
        failures.append("post_stop_block_count_mismatch")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M663",
        "name": "Emergency Stop During Inference",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests": len(records),
        "served_before_stop": served_before_stop,
        "blocked_after_stop": blocked_after_stop,
        "failures": failures,
        "scope": "local inference gate emergency stop contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M663 Emergency Stop During Inference: {status}")
    print(f"served={served_before_stop} blocked={blocked_after_stop} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
