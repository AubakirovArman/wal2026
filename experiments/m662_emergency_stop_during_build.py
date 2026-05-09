from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m662_emergency_stop_during_build_results.json"


def build_worker(stop_event: threading.Event, progress: list[int]) -> None:
    for step in range(100):
        if stop_event.is_set():
            break
        progress.append(step)
        time.sleep(0.001)


def main() -> int:
    stop_event = threading.Event()
    progress: list[int] = []
    worker = threading.Thread(target=build_worker, args=(stop_event, progress))
    worker.start()
    time.sleep(0.01)
    stop_event.set()
    worker.join(timeout=2)
    stopped_before_complete = len(progress) < 100
    thread_stopped = not worker.is_alive()
    status = "PASS" if stopped_before_complete and thread_stopped else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M662",
        "name": "Emergency Stop During Build",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps_completed": len(progress),
        "stopped_before_complete": stopped_before_complete,
        "thread_stopped": thread_stopped,
        "scope": "local build-loop emergency stop contract",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M662 Emergency Stop During Build: {status}")
    print(f"steps_completed={len(progress)} thread_stopped={thread_stopped}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
