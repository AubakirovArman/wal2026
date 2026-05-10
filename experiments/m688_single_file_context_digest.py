from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = ROOT / "WAL_AIGI_FULL_CONTEXT.md"
RESULT_PATH = ROOT / "experiments" / "m688_single_file_context_digest_results.json"


REQUIRED_SECTIONS = [
    "Short Positioning",
    "Current Metrics",
    "Repository Map",
    "WAL Architecture",
    "AIGI 1.0 Architecture",
    "AIGI Gates M679-M697",
    "Current Validation Ledger",
    "Status Semantics",
    "Controlled Runner Taxonomy",
    "Legacy Audit",
    "Small-Model Status",
    "Important Commands",
    "Known Limitations / Non-Claims",
    "Recommended Next Steps",
    "Canonical Source Files",
    "One-Line Summary",
]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_SUMMARY.md",
    "TECHNICAL_REPORT.md",
    "KNOWN_ISSUES.md",
    "docs/VALIDATION_STATUS.md",
    "docs/project_metrics.json",
    "docs/dev_diary_ru.md",
    "docs/aigi/README.md",
    "docs/aigi/dev_diary_ru.md",
    "docs/aigi/test_log.md",
    "experiments/experiments_manifest.json",
    "experiments/m625_safe_runtime_sweep_results.json",
    "experiments/m631_docs_command_smoke_results.json",
    "site/index.html",
]

FORBIDDEN_PATTERNS = {
    "production_ready": re.compile(r"\bproduction[- ]ready\b", re.IGNORECASE),
    "certified_a_plus": re.compile(r"certified\s+A\+", re.IGNORECASE),
    "complete_and_production": re.compile(r"complete\s+and\s+production", re.IGNORECASE),
}


def main() -> int:
    metrics = json.loads((ROOT / "docs" / "project_metrics.json").read_text(encoding="utf-8"))
    text = CONTEXT_PATH.read_text(encoding="utf-8") if CONTEXT_PATH.exists() else ""
    checks = []

    checks.append({"name": "context_file_exists", "passed": CONTEXT_PATH.exists()})
    checks.append({"name": "line_count_at_least_250", "passed": len(text.splitlines()) >= 250})
    for section in REQUIRED_SECTIONS:
        checks.append({"name": f"section:{section}", "passed": section in text})
    for rel_path in REQUIRED_FILES:
        checks.append({"name": f"source_exists:{rel_path}", "passed": (ROOT / rel_path).exists()})

    expected_values = {
        "milestone_scripts": metrics["experiments"]["milestone_scripts"],
        "python_scripts_total": metrics["experiments"]["python_scripts_total"],
        "result_json_files": metrics["experiments"]["result_json_files"],
        "book_entries": metrics["documentation"]["book_entries"],
        "pytest_tests": metrics["code"]["pytest_tests"],
        "safe_sweep_pass": metrics["experiments"]["safe_sweep_pass"],
        "safe_sweep_blocked": metrics["experiments"]["safe_sweep_blocked"],
        "schema_valid": metrics["experiments"]["result_json_files"],
        "docs_smoke": 69,
    }
    checks.extend([
        {"name": "metric:milestone_scripts", "passed": f"Milestone scripts | {expected_values['milestone_scripts']}" in text},
        {"name": "metric:python_scripts_total", "passed": f"Python scripts in `experiments/` | {expected_values['python_scripts_total']}" in text},
        {"name": "metric:result_json_files", "passed": f"Result JSON files | {expected_values['result_json_files']}" in text},
        {"name": "metric:book_entries", "passed": f"Book entries | {expected_values['book_entries']}" in text},
        {"name": "metric:pytest_tests", "passed": f"Maintained pytest tests | {expected_values['pytest_tests']}" in text},
        {"name": "metric:safe_sweep", "passed": f"Safe sweep | {expected_values['safe_sweep_pass']} PASS / {expected_values['safe_sweep_blocked']} BLOCKED" in text},
        {"name": "metric:schema", "passed": f"Result schema | {expected_values['schema_valid']} valid / 0 invalid" in text},
        {"name": "metric:docs_smoke", "passed": f"Docs smoke | {expected_values['docs_smoke']} / {expected_values['docs_smoke']} commands PASS" in text},
    ])

    for name, pattern in FORBIDDEN_PATTERNS.items():
        checks.append({"name": f"forbidden:{name}", "passed": pattern.search(text) is None})

    checks.append({"name": "non_claim:autonomous_agi", "passed": "not autonomous agi" in text.lower() or "**not** autonomous agi" in text.lower()})
    lowered = text.lower()
    checks.append({
        "name": "non_claim:semantic_weight_editing",
        "passed": "not a base-weight semantic editing backend" in lowered
        or "does not perform lora/memit base-weight semantic editing" in lowered,
    })
    checks.append({"name": "aigi_latest_range", "passed": "M679-M697" in text})
    checks.append({"name": "single_file_summary", "passed": "single-file operational context" in text})

    failures = [check for check in checks if not check["passed"]]
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M688",
        "name": "Single File Context Digest",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failures": failures,
        "artifact": str(CONTEXT_PATH.relative_to(ROOT)),
        "docs": "WAL_AIGI_FULL_CONTEXT.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M688 Single File Context Digest: {status}")
    print(f"checks={len(checks) - len(failures)}/{len(checks)} artifact={CONTEXT_PATH.relative_to(ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
