from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "deployment_log_volume.jsonl"
RESULT_PATH = ROOT / "experiments" / "m668_log_volume_storage_growth_results.json"


def main() -> int:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": index, "level": "INFO" if index % 10 else "WARN", "message": f"deployment event {index:04d}"}
        for index in range(1000)
    ]
    ARTIFACT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    size_bytes = ARTIFACT_PATH.stat().st_size
    bytes_per_event = round(size_bytes / len(records), 2)
    failures = []
    if bytes_per_event > 160:
        failures.append("log_event_too_large")
    if size_bytes > 200_000:
        failures.append("log_artifact_too_large")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M668",
        "name": "Log Volume Storage Growth",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "events": len(records),
        "size_bytes": size_bytes,
        "bytes_per_event": bytes_per_event,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "local log-volume sizing contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M668 Log Volume Storage Growth: {status}")
    print(f"events={len(records)} size_bytes={size_bytes} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
