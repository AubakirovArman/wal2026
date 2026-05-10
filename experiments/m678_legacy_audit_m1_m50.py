from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wal.legacy_audit import build_manifest  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m678_legacy_audit_m1_m50_results.json"
AUDIT_PATH = ROOT / "experiments" / "legacy_audit_m1_m50.json"
DOC_PATH = ROOT / "docs" / "legacy_audit_m1_m50.md"


def render_doc(audit: dict[str, object]) -> str:
    summary = audit["summary"]
    records = audit["records"]
    lines = [
        "# Legacy Audit M1-M50",
        "",
        "Date: 2026-05-10",
        "",
        "This is the first Legacy Experiment Resurrection batch. It audits scripts with numeric prefixes M1-M50 using modern M624/M625 safety metadata, source signals, and historical artifact discovery.",
        "",
        "## Summary",
        "",
        f"- Scripts audited: `{summary['total']}`",
        f"- Current public claim allowed after audit: `{summary['public_claim_allowed']}`",
        f"- With historical artifacts: `{summary['with_artifacts']}`",
        f"- With schema-v1 artifacts: `{summary['with_schema_v1_artifact']}`",
        "",
        "## Review Status Counts",
        "",
    ]
    for status, count in summary["by_review_status"].items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Runner Type Counts", ""])
    for runner, count in summary["by_runner_type"].items():
        lines.append(f"- `{runner}`: `{count}`")
    lines.extend([
        "",
        "## Highest Priority Modernization Items",
        "",
    ])
    recommendation_counts: dict[str, int] = {}
    for record in records:
        for recommendation in record["modernization_recommendations"]:
            recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
    for recommendation, count in sorted(recommendation_counts.items(), key=lambda item: (-item[1], item[0]))[:12]:
        lines.append(f"- `{recommendation}`: `{count}`")
    lines.extend([
        "",
        "## Per-Script Review",
        "",
        "| File | Runner | Review Status | Artifacts | Key Fixes |",
        "|------|--------|---------------|-----------|-----------|",
    ])
    for record in records:
        fixes = ", ".join(record["modernization_recommendations"][:4]) or "none"
        lines.append(
            f"| `{record['file']}` | `{record['runner_type']}` | `{record['review_status']}` | "
            f"`{record['artifact_count']}` | {fixes} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "M1-M50 are mostly core WAL encoding/runtime experiments. Safe-pass scripts are not upgraded to current public claims until they emit schema-v1 results and clearer hardware/model metadata.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    audit = build_manifest(ROOT, lower=1, upper=50)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOC_PATH.write_text(render_doc(audit), encoding="utf-8")
    summary = audit["summary"]
    failures: list[str] = []
    if summary["total"] == 0:
        failures.append("empty_batch")
    if "invalid_due_to_source_error" in summary["by_review_status"]:
        failures.append("source_errors_in_batch")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M678",
        "name": "Legacy Audit M1-M50",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit": str(AUDIT_PATH.relative_to(ROOT)),
        "docs": str(DOC_PATH.relative_to(ROOT)),
        "summary": summary,
        "failures": failures,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M678 Legacy Audit M1-M50: {status}")
    print(f"scripts={summary['total']} statuses={summary['by_review_status']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
