from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "deployment_canary_traffic.json"
RESULT_PATH = ROOT / "experiments" / "m660_canary_real_traffic_simulation_results.json"


def route_request(index: int, canary_percent: int) -> str:
    return "canary" if index % 100 < canary_percent else "stable"


def main() -> int:
    canary_percent = 10
    requests = [{"id": index, "route": route_request(index, canary_percent)} for index in range(1000)]
    canary = [request for request in requests if request["route"] == "canary"]
    stable = [request for request in requests if request["route"] == "stable"]
    observed = round(len(canary) / len(requests), 3)
    failures = []
    if observed != 0.1:
        failures.append("canary_ratio_mismatch")
    if not stable:
        failures.append("stable_route_empty")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps({"canary_percent": canary_percent, "observed": observed, "sample": requests[:20]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M660",
        "name": "Canary Real Traffic Simulation",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests": len(requests),
        "canary_requests": len(canary),
        "stable_requests": len(stable),
        "observed_canary_ratio": observed,
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "deterministic local traffic router simulation",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M660 Canary Real Traffic Simulation: {status}")
    print(f"requests={len(requests)} canary={len(canary)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
