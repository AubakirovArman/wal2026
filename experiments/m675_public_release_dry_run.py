from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "product_public_release_dry_run.json"
RESULT_PATH = ROOT / "experiments" / "m675_public_release_dry_run_results.json"


REQUIRED_RESULTS = [
    "m621_release_truthfulness_audit_results.json",
    "m622_result_schema_gate_results.json",
    "m623_core_release_gate_results.json",
    "m624_full_test_inventory_results.json",
    "m630_public_claim_checker_results.json",
    "m631_docs_command_smoke_results.json",
    "m673_demo_script_e2e_results.json",
    "m674_github_pages_build_results.json",
]

REQUIRED_DOCS = [
    "README.md",
    "TECHNICAL_REPORT.md",
    "PROJECT_SUMMARY.md",
    "docs/product_polish_protocol.md",
    "site/index.html",
]


def main() -> int:
    records = []
    failures = []
    for result_name in REQUIRED_RESULTS:
        path = ROOT / "experiments" / result_name
        if not path.exists():
            failures.append({"target": result_name, "reason": "missing_result"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        allowed = status == "PASS"
        records.append({"target": result_name, "status": status, "allowed": allowed})
        if not allowed:
            failures.append({"target": result_name, "reason": f"unexpected_status:{status}"})
    for rel_path in REQUIRED_DOCS:
        if not (ROOT / rel_path).exists():
            failures.append({"target": rel_path, "reason": "missing_doc_or_site"})

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "pre-alpha" not in readme.lower():
        failures.append({"target": "README.md", "reason": "missing_pre_alpha"})

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"records": records, "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M675",
        "name": "Public Release Dry Run",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_checked": len(records),
        "docs_checked": len(REQUIRED_DOCS),
        "failures": failures,
        "self_referential_note": "M625 safe sweep is validated outside M675 to avoid a cyclic gate dependency.",
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "pre-alpha public release dry run gate",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M675 Public Release Dry Run: {status}")
    print(f"results={len(records)} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
