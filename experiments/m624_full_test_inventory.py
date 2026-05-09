"""M624 — Full Test Inventory.

Builds an ordered inventory of every experiment/test script and classifies
whether it is safe to execute in an automated local sweep.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

HEAVY_PATTERNS = {
    "/mnt/hf_model_weights": "local_model_path",
    "from_pretrained": "model_load",
    "AutoModel": "model_load",
    "AutoTokenizer": "tokenizer_load",
    "torch.cuda": "cuda",
    ".cuda(": "cuda",
    "cuda:": "cuda",
    "device_map": "device_map",
    "triton": "triton",
    "load_dataset": "dataset_load",
    "datasets": "dataset_load",
    "hf_hub_download": "hf_download",
    ".safetensors": "model_artifact",
}

UNSAFE_PATTERNS = {
    "git commit": "git_mutation",
    "git tag": "git_mutation",
    "git checkout": "git_mutation",
    "git merge": "git_mutation",
    "shutil.rmtree": "destructive_file_op",
    "os.remove": "destructive_file_op",
    ".unlink(": "destructive_file_op",
    "rm -rf": "destructive_shell_op",
    "generate_detailed_book": "mass_regeneration",
    "generate_detailed_diary": "mass_regeneration",
    "license_header_injection": "mass_rewrite",
}

UNSAFE_NAME_FRAGMENTS = {
    "license_header_injection": "mass_rewrite",
    "generator": "public_doc_generator",
    "generate_": "public_doc_generator",
    "readme_updater": "public_claim_generator",
    "readme_generator": "public_claim_generator",
    "readme_badges": "public_claim_generator",
    "project_readme": "public_claim_generator",
    "badge": "public_claim_generator",
    "final_declaration": "public_claim_generator",
    "final_commit": "git_metadata_generator",
    "final_export": "mass_export",
    "final_report": "public_claim_generator",
    "final_html_report": "public_claim_generator",
    "final_status_dashboard": "public_claim_generator",
    "final_validation_suite": "public_claim_generator",
    "release_notes": "public_claim_generator",
    "project_summary": "public_claim_generator",
    "project_manifest": "public_claim_generator",
    "project_audit": "public_claim_generator",
    "project_growth": "public_claim_generator",
    "project_roadmap": "public_doc_generator",
    "project_stats": "public_claim_generator",
    "project_index": "public_doc_generator",
    "project_inventory": "public_doc_generator",
    "project_sitemap": "public_doc_generator",
    "project_glossary": "public_doc_generator",
    "project_faq": "public_doc_generator",
    "project_todo": "public_doc_generator",
    "project_acknowledgments": "public_doc_generator",
    "project_lessons": "public_doc_generator",
    "project_retrospective": "public_doc_generator",
    "project_metrics": "public_claim_generator",
    "project_kpis": "public_claim_generator",
    "project_scorecard": "public_claim_generator",
    "project_completion": "public_claim_generator",
    "project_certification": "public_claim_generator",
    "completion_certificate": "public_claim_generator",
    "publication_readiness": "public_claim_generator",
    "contribution_guide": "public_doc_generator",
    "project_guidelines": "public_doc_generator",
    "project_standards": "public_doc_generator",
    "project_policies": "public_doc_generator",
    "milestone": "public_claim_generator",
    "wrap_up": "public_claim_generator",
    "book_consolidation": "mass_regeneration",
    "doc_generator": "mass_regeneration",
    "changelog_generator": "public_claim_generator",
    "contributing": "public_doc_generator",
    "security_policy": "public_doc_generator",
    "code_of_conduct": "public_doc_generator",
    "issue_template": "public_doc_generator",
    "pr_template": "public_doc_generator",
    "citation_bibtex": "public_doc_generator",
    "wal_studio_readme": "public_doc_generator",
    "final_statistics": "public_claim_generator",
    "executive_summary": "public_claim_generator",
    "health_score": "public_claim_generator",
    "final_dashboard": "public_claim_generator",
    "project_archive": "archive_mutation",
    "backup": "archive_mutation",
    "restore": "archive_mutation",
    "git_": "git_mutation",
    "github_repo_init": "git_mutation",
    "branch": "git_mutation",
    "merge": "merge_simulation_or_mutation",
    "cleanup": "destructive_file_op",
    "pruning": "destructive_file_op",
    "archiving": "archive_mutation",
    "export": "mass_export",
    "m31_sparse_probe": "runtime_timeout_in_safe_sweep",
    "m37_entropy_regularized_encoder": "runtime_timeout_in_safe_sweep",
    "m39_hybrid_encoder": "runtime_timeout_in_safe_sweep",
}

SAFE_SUBPROCESS_ALLOWLIST = {
    "m518_automated_test_suite.py",
    "m623_core_release_gate.py",
    "m631_docs_command_smoke.py",
}

SAFE_TEXT_ONLY_AUDIT_ALLOWLIST = {
    "m628_blocked_script_taxonomy.py",
    "m629_controlled_runner_matrix.py",
}


def order_key(path: Path) -> tuple[int, str, str]:
    match = re.match(r"m(\d+)([a-z]*)_", path.name)
    if match:
        return int(match.group(1)), match.group(2), path.name
    return 999999, "", path.name


def classify(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    reasons: list[str] = []
    parse_status = "PASS"
    parse_error = None
    try:
        compile(text, str(path), "exec")
    except SyntaxError as exc:
        parse_status = "FAIL"
        parse_error = f"{exc.msg} at line {exc.lineno}"
        reasons.append("syntax_error")

    lowered = text.lower()
    lowered_name = path.name.lower()
    for fragment, reason in UNSAFE_NAME_FRAGMENTS.items():
        if fragment in lowered_name and reason not in reasons:
            reasons.append(reason)
    for needle, reason in HEAVY_PATTERNS.items():
        if needle.lower() in lowered and reason not in reasons:
            reasons.append(reason)
    for needle, reason in UNSAFE_PATTERNS.items():
        if needle.lower() in lowered and reason not in reasons:
            reasons.append(reason)
    if "subprocess" in lowered and path.name not in SAFE_SUBPROCESS_ALLOWLIST:
        reasons.append("subprocess")

    if path.name in SAFE_TEXT_ONLY_AUDIT_ALLOWLIST:
        reasons = [reason for reason in reasons if reason == "syntax_error"]

    runnable = parse_status == "PASS" and not reasons
    if path.name in {"m624_full_test_inventory.py", "m625_safe_runtime_sweep.py"}:
        runnable = False
        reasons.append("self_referential_audit_script")

    return {
        "file": path.name,
        "order": order_key(path)[:2],
        "parse_status": parse_status,
        "parse_error": parse_error,
        "runnable": runnable,
        "blocked_reasons": sorted(set(reasons)),
    }


def main() -> None:
    records = [classify(path) for path in sorted(EXPERIMENTS.glob("*.py"), key=order_key)]
    parse_failures = [record for record in records if record["parse_status"] != "PASS"]
    runnable = [record for record in records if record["runnable"]]
    blocked = [record for record in records if not record["runnable"]]
    reason_counts: dict[str, int] = {}
    for record in blocked:
        for reason in record["blocked_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    result = {
        "schema_version": "wal.results.v1",
        "status": "PASS" if not parse_failures else "FAIL",
        "pass": not parse_failures,
        "total_scripts": len(records),
        "parse_failures": len(parse_failures),
        "runnable_scripts": len(runnable),
        "blocked_scripts": len(blocked),
        "blocked_reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }

    print("=" * 60)
    print("M624 — FULL TEST INVENTORY")
    print("=" * 60)
    print(f"  Scripts: {len(records)}")
    print(f"  Parse failures: {len(parse_failures)}")
    print(f"  Runnable: {len(runnable)}")
    print(f"  Blocked: {len(blocked)}")

    (EXPERIMENTS / "m624_full_test_inventory_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nM624 status={result['status']}")


if __name__ == "__main__":
    main()
