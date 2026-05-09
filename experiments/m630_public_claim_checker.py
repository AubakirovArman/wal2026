from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m630_public_claim_checker_results.json"
DOC_PATH = ROOT / "docs" / "public_claim_policy.md"


PUBLIC_FILES = [
    "README.md",
    "PROJECT_SUMMARY.md",
    "PROJECT_SUMMARY_v2.json",
    "TECHNICAL_REPORT.md",
    "docs/demo_playbook.md",
    "docs/blocked_script_taxonomy.md",
    "docs/controlled_runners.md",
    "docs/public_claim_policy.md",
    "docs/docs_command_smoke.md",
    "docs/model_small_protocol.md",
    "docs/cross_model_validation_plan.md",
    "docs/robustness_data_protocol.md",
    "docs/ci_hardening_protocol.md",
    "docs/wal_status_summary.md",
    "wal_studio_v01/README.md",
    "FINAL_REPORT.html",
    "FINAL_REPORT.json",
    "MANIFEST.json",
    "MILESTONE_v1.0.json",
    "MILESTONE_v1.2.json",
    "MILESTONE_v1.4.json",
    "WAL_EXPORT.json",
    "RELEASE_NOTES_v2.md",
    "FAQ.md",
]

FORBIDDEN_PATTERNS = {
    "affirmative_production_ready": re.compile(r"\bproduction[- ]ready\b|PRODUCTION READY", re.IGNORECASE),
    "active_top_grade": re.compile(r'"grade"\s*:\s*"A\+"|>A\+<|certified-A\+', re.IGNORECASE),
    "active_certification": re.compile(r"certified\s+A\+|externally certified:\s*true", re.IGNORECASE),
    "complete_production_claim": re.compile(r"complete\s+and\s+production", re.IGNORECASE),
}

REQUIRED_PHRASES = {
    "README.md": ["pre-alpha", "TECHNICAL_REPORT.md", "docs/demo_playbook.md"],
    "TECHNICAL_REPORT.md": ["pre-alpha research framework", "Limitations", "Recommended Public Claims"],
    "docs/demo_playbook.md": ["pre-alpha", "BLOCKED", "UNSUPPORTED"],
}


def render_policy(result: dict[str, object]) -> str:
    lines = [
        "# Public Claim Policy",
        "",
        "Date: 2026-05-09",
        "",
        "## Purpose",
        "",
        "Public docs must describe WAL as a pre-alpha research framework unless a later gate proves a stronger claim.",
        "",
        "## Allowed Claims",
        "",
        "- pre-alpha research framework",
        "- research-grade prototype",
        "- safe-core validated",
        "- schema-valid result corpus",
        "- deployment simulation or prototype when not separately validated",
        "",
        "## Blocked Claim Types",
        "",
        "- Mature deployment readiness claims.",
        "- Active top-grade release labels.",
        "- External certification claims without a real external audit.",
        "- Blanket claims that every historical module is a real-world validation.",
        "",
        "## Current Scan",
        "",
        f"- Files scanned: `{result['files_scanned']}`",
        f"- Violations: `{result['violations_total']}`",
        f"- Required phrase misses: `{result['required_phrase_misses']}`",
        "",
        "## Gate",
        "",
        "M630 must pass before accepting generated README, badge, release-note, final-report, or milestone artifacts.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    violations: list[dict[str, str]] = []
    missing_required: list[dict[str, str]] = []
    scanned = []

    for rel_path in PUBLIC_FILES:
        path = ROOT / rel_path
        if not path.exists():
            missing_required.append({"file": rel_path, "missing": "file"})
            continue
        text = path.read_text(encoding="utf-8")
        scanned.append(rel_path)
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                violations.append({"file": rel_path, "pattern": name})
        for phrase in REQUIRED_PHRASES.get(rel_path, []):
            if phrase.lower() not in text.lower():
                missing_required.append({"file": rel_path, "missing": phrase})

    status = "PASS" if not violations and not missing_required else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M630",
        "name": "Public Claim Checker",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_scanned": len(scanned),
        "violations_total": len(violations),
        "required_phrase_misses": len(missing_required),
        "violations": violations,
        "missing_required": missing_required,
        "docs": str(DOC_PATH.relative_to(ROOT)),
    }

    DOC_PATH.write_text(render_policy(result), encoding="utf-8")
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M630 Public Claim Checker: {status}")
    print(f"files={len(scanned)} violations={len(violations)} missing={len(missing_required)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
