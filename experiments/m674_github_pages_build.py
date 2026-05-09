from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
RESULT_PATH = ROOT / "experiments" / "m674_github_pages_build_results.json"


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = (ROOT / "PROJECT_SUMMARY.md").read_text(encoding="utf-8")
    SITE.mkdir(parents=True, exist_ok=True)
    index = SITE / "index.html"
    status_json = SITE / "status.json"
    body = "\n".join([
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>WAL Pre-alpha</title></head><body>",
        "<h1>WAL — WeightOps Research Framework</h1>",
        "<p>Status: pre-alpha research framework prototype.</p>",
        "<h2>README</h2>",
        f"<pre>{html.escape(readme[:6000])}</pre>",
        "<h2>Project Summary</h2>",
        f"<pre>{html.escape(summary[:4000])}</pre>",
        "</body></html>",
    ])
    index.write_text(body, encoding="utf-8")
    status_payload = {
        "status": "pre-alpha research framework prototype",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": ["README.md", "PROJECT_SUMMARY.md"],
    }
    status_json.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = []
    if "production-ready" in body.lower():
        failures.append("forbidden_claim_in_site")
    if not index.exists() or not status_json.exists():
        failures.append("site_artifact_missing")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M674",
        "name": "GitHub Pages Build",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "site_files": [str(index.relative_to(ROOT)), str(status_json.relative_to(ROOT))],
        "failures": failures,
        "scope": "static local GitHub Pages artifact build",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M674 GitHub Pages Build: {status}")
    print(f"files={len(result['site_files'])} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
