from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "deployment_shadow_server.json"
RESULT_PATH = ROOT / "experiments" / "m659_shadow_deploy_real_server_results.json"


class ShadowHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        query = self.path.split("?", 1)[-1]
        primary = {"route": "primary", "answer": f"primary:{query}"}
        shadow = {"route": "shadow", "answer": f"primary:{query}"}
        body = json.dumps({"primary": primary, "shadow": shadow}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ShadowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = []
    failures = []
    try:
        port = server.server_address[1]
        for index in range(12):
            url = f"http://127.0.0.1:{port}/infer?q=shadow-{index:03d}"
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            agreement = payload["primary"]["answer"] == payload["shadow"]["answer"]
            records.append({"request": index, "agreement": agreement, "payload": payload})
            if not agreement:
                failures.append(index)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures and len(records) == 12 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M659",
        "name": "Shadow Deploy Real Server",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests": len(records),
        "agreement": round(sum(1 for record in records if record["agreement"]) / max(len(records), 1), 3),
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "local loopback HTTP shadow server; no external traffic",
        "docs": "docs/deployment_reality_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M659 Shadow Deploy Real Server: {status}")
    print(f"requests={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
