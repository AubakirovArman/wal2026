from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "m631_docs_command_smoke_results.json"
DOC_PATH = ROOT / "docs" / "docs_command_smoke.md"


RUN_COMMANDS = [
    "PYTHONPATH=src python -m pytest -q tests",
    "PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid",
    "python experiments/m626_technical_report.py",
    "python experiments/m627_polished_demo_playbook.py",
    "python experiments/m628_blocked_script_taxonomy.py",
    "python experiments/m629_controlled_runner_matrix.py",
    "python experiments/m630_public_claim_checker.py",
    "python experiments/m632_llama_1b_full_workflow.py",
    "python experiments/m633_qwen_small_full_workflow.py",
    "python experiments/m634_gemma_small_full_workflow.py",
    "python experiments/m635_tinyllama_mistral_full_workflow.py",
    "python experiments/m636_cross_model_recipe_replay.py",
    "python experiments/m637_cross_model_layer_aperture.py",
    "python experiments/m638_cross_model_ci_behavior.py",
    "python experiments/m639_dirty_facts_corpus.py",
    "python experiments/m640_ambiguous_facts_test.py",
    "python experiments/m641_temporal_facts_date_logic.py",
    "python experiments/m642_long_answer_facts.py",
    "python experiments/m643_procedural_knowledge_routing.py",
    "python experiments/m644_policy_refusal_edits.py",
    "python experiments/m645_hard_facts_hybrid_backend.py",
    "python experiments/m646_negative_test_expansion.py",
    "python experiments/m647_lure_test_expansion.py",
    "python experiments/m648_context_stress_8k_32k.py",
    "python experiments/m649_auto_test_quality_audit.py",
    "python experiments/m650_ci_score_calibration.py",
    "python experiments/m651_behavioral_checksum_drift.py",
    "python experiments/m652_recipe_secret_scanner.py",
    "python experiments/m653_malicious_recipe_injection.py",
    "python experiments/m654_registry_poisoning_test.py",
    "python experiments/m655_hotfix_abuse_test.py",
    "python experiments/m656_prompt_injection_retrieval_context.py",
    "python experiments/m657_provenance_tamper_test.py",
    "python experiments/m658_signed_package_verification.py",
    "python wal_studio_v01/demo.py",
]

EXISTS_ONLY_COMMANDS = [
    "python experiments/m624_full_test_inventory.py",
    "python experiments/m625_safe_runtime_sweep.py --timeout 15",
]

DOC_FILES = [
    "README.md",
    "TECHNICAL_REPORT.md",
    "docs/demo_playbook.md",
    "docs/controlled_runners.md",
    "docs/model_small_protocol.md",
    "docs/cross_model_validation_plan.md",
    "docs/robustness_data_protocol.md",
    "docs/ci_hardening_protocol.md",
    "docs/security_hardening_protocol.md",
]


def command_exists(command: str) -> bool:
    parts = command.split()
    if "python" not in parts:
        return True
    for part in parts:
        if part.endswith(".py"):
            return (ROOT / part).exists()
    return True


def result_path_for_command(command: str) -> Path | None:
    for part in command.split():
        if part.startswith("experiments/") and part.endswith(".py"):
            path = ROOT / part
            return path.with_name(path.stem + "_results.json")
    return None


def run_command(command: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}".rstrip(":")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    output = completed.stdout[-4000:]
    record = {
        "command": command,
        "mode": "run",
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output_tail": output,
    }
    result_path = result_path_for_command(command)
    if result_path and result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            record["result_status"] = payload.get("status")
            record["result_pass"] = payload.get("pass")
            record["result_file"] = str(result_path.relative_to(ROOT))
        except Exception as exc:
            record["result_status_error"] = str(exc)
    return record


def render_doc(result: dict[str, object]) -> str:
    lines = [
        "# Docs Command Smoke",
        "",
        "Date: 2026-05-09",
        "",
        "## Purpose",
        "",
        "Verify that the fast public documentation commands still run, while long sweep commands are checked for target existence and kept as explicit reviewer commands.",
        "",
        "## Results",
        "",
        f"- Runnable commands: `{result['run_commands']}`",
        f"- Runnable commands passed: `{result['run_passed']}`",
        f"- Exists-only commands: `{result['exists_only_commands']}`",
        f"- Exists-only commands passed: `{result['exists_only_passed']}`",
        "",
        "## Commands",
        "",
        f"- Commands with embedded blocked result status: `{result['blocked_result_commands']}`",
        "",
        "| Mode | Command | Command Status | Result Status |",
        "|------|---------|----------------|---------------|",
    ]
    for record in result["records"]:
        status = "PASS" if record["passed"] else "FAIL"
        result_status = record.get("result_status", "—")
        lines.append(f"| `{record['mode']}` | `{record['command']}` | `{status}` | `{result_status}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    records: list[dict[str, object]] = []

    for rel_path in DOC_FILES:
        records.append({
            "command": f"doc exists: {rel_path}",
            "mode": "doc",
            "passed": (ROOT / rel_path).exists(),
            "returncode": 0 if (ROOT / rel_path).exists() else 1,
        })

    for command in RUN_COMMANDS:
        records.append(run_command(command))

    for command in EXISTS_ONLY_COMMANDS:
        exists = command_exists(command)
        records.append({
            "command": command,
            "mode": "exists_only",
            "passed": exists,
            "returncode": 0 if exists else 1,
        })

    failed = [record for record in records if not record["passed"]]
    run_records = [record for record in records if record["mode"] == "run"]
    exists_records = [record for record in records if record["mode"] == "exists_only"]
    blocked_result_commands = sum(1 for record in records if record.get("result_status") == "BLOCKED")
    status = "PASS" if not failed else "FAIL"

    result = {
        "schema_version": "wal.results.v1",
        "module": "M631",
        "name": "Docs Command Smoke",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_commands": len(run_records),
        "run_passed": sum(1 for record in run_records if record["passed"]),
        "exists_only_commands": len(exists_records),
        "exists_only_passed": sum(1 for record in exists_records if record["passed"]),
        "blocked_result_commands": blocked_result_commands,
        "docs_checked": len(DOC_FILES),
        "failures": failed,
        "records": records,
        "docs": str(DOC_PATH.relative_to(ROOT)),
    }

    DOC_PATH.write_text(render_doc(result), encoding="utf-8")
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M631 Docs Command Smoke: {status}")
    print(f"run={result['run_passed']}/{result['run_commands']} exists={result['exists_only_passed']}/{result['exists_only_commands']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
