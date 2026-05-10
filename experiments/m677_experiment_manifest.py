from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wal.legacy_audit import build_manifest  # noqa: E402


RESULT_PATH = ROOT / "experiments" / "m677_experiment_manifest_results.json"
MANIFEST_PATH = ROOT / "experiments" / "experiments_manifest.json"
DOC_PATH = ROOT / "docs" / "legacy_audit_manifest.md"


def render_doc(manifest: dict[str, object]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Legacy Experiment Manifest",
        "",
        "Date: 2026-05-10",
        "",
        "This manifest classifies every `experiments/*.py` script into a runner type and a modern review status.",
        "",
        "## Summary",
        "",
        f"- Total scripts: `{summary['total']}`",
        f"- With historical artifacts: `{summary['with_artifacts']}`",
        f"- With `wal.results.v1` artifacts: `{summary['with_schema_v1_artifact']}`",
        f"- Current public claim allowed after audit: `{summary['public_claim_allowed']}`",
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
        "## Policy",
        "",
        "- `still_valid` means a script passed the modern safe sweep and has a schema-v1 result artifact.",
        "- `still_valid_needs_schema_v1` means the script still runs but is not allowed as a current public claim until it emits `wal.results.v1`.",
        "- Blocked statuses are not failures; they require controlled runners such as GPU/model, slow, mutation dry-run, or subprocess review.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = build_manifest(ROOT)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DOC_PATH.write_text(render_doc(manifest), encoding="utf-8")
    summary = manifest["summary"]
    failures: list[str] = []
    if summary["total"] < 800:
        failures.append("unexpected_script_count")
    if not summary["by_review_status"]:
        failures.append("missing_review_status_counts")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M677",
        "name": "Experiment Manifest",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "docs": str(DOC_PATH.relative_to(ROOT)),
        "summary": summary,
        "failures": failures,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M677 Experiment Manifest: {status}")
    print(f"scripts={summary['total']} review_statuses={len(summary['by_review_status'])}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
