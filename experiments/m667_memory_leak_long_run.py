from __future__ import annotations

import gc
import json
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m667_memory_leak_long_run_results.json"


def main() -> int:
    tracemalloc.start()
    snapshots = []
    for index in range(200):
        payload = [f"request-{index}-{item}" for item in range(50)]
        if index % 20 == 0:
            current, peak = tracemalloc.get_traced_memory()
            snapshots.append({"iteration": index, "current_bytes": current, "peak_bytes": peak})
        del payload
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    short_sentinel_passed = current < peak and peak < 2_000_000
    status = "SIMULATED" if short_sentinel_passed else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M667",
        "name": "Memory Leak Long Run",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": 200,
        "current_bytes": current,
        "peak_bytes": peak,
        "short_sentinel_passed": short_sentinel_passed,
        "snapshots": snapshots,
        "scope": "short memory sentinel only; not a true long-run service test",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M667 Memory Leak Long Run: {status}")
    print(f"iterations=200 peak_bytes={peak}")
    return 0 if status in {"SIMULATED", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
