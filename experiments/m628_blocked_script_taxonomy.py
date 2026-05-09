from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "experiments" / "m624_full_test_inventory_results.json"
RESULT_PATH = ROOT / "experiments" / "m628_blocked_script_taxonomy_results.json"
DOC_PATH = ROOT / "docs" / "blocked_script_taxonomy.md"


REASON_TO_RUNNER = {
    "cuda": "GPU_HEAVY",
    "triton": "GPU_HEAVY",
    "device_map": "GPU_HEAVY",
    "local_model_path": "GPU_HEAVY",
    "model_artifact": "GPU_HEAVY",
    "model_load": "MODEL_CONTROLLED",
    "tokenizer_load": "MODEL_CONTROLLED",
    "dataset_load": "MODEL_CONTROLLED",
    "hf_download": "MODEL_CONTROLLED",
    "git_mutation": "MUTATION_DRY_RUN",
    "git_metadata_generator": "MUTATION_DRY_RUN",
    "merge_simulation_or_mutation": "MUTATION_DRY_RUN",
    "destructive_file_op": "MUTATION_DRY_RUN",
    "destructive_shell_op": "MUTATION_DRY_RUN",
    "archive_mutation": "MUTATION_DRY_RUN",
    "public_claim_generator": "DOCS_PUBLIC_CLAIMS",
    "public_doc_generator": "DOCS_PUBLIC_CLAIMS",
    "mass_regeneration": "DOCS_PUBLIC_CLAIMS",
    "mass_rewrite": "DOCS_PUBLIC_CLAIMS",
    "mass_export": "DOCS_PUBLIC_CLAIMS",
    "runtime_timeout_in_safe_sweep": "SLOW_PROFILE",
    "subprocess": "SUBPROCESS_REVIEW",
    "self_referential_audit_script": "INTERNAL_AUDIT",
    "syntax_error": "BROKEN_COMPILE",
}

PRIMARY_RUNNER_PRIORITY = [
    "BROKEN_COMPILE",
    "MUTATION_DRY_RUN",
    "DOCS_PUBLIC_CLAIMS",
    "GPU_HEAVY",
    "MODEL_CONTROLLED",
    "SLOW_PROFILE",
    "SUBPROCESS_REVIEW",
    "INTERNAL_AUDIT",
]


def classify_record(record: dict[str, object]) -> dict[str, object]:
    reasons = list(record.get("blocked_reasons") or [])
    runners = sorted({REASON_TO_RUNNER.get(reason, "UNASSIGNED") for reason in reasons})
    primary = next(
        (runner for runner in PRIMARY_RUNNER_PRIORITY if runner in runners),
        runners[0] if runners else "UNASSIGNED",
    )
    return {
        "file": record["file"],
        "blocked_reasons": reasons,
        "candidate_runners": runners,
        "primary_runner": primary,
    }


def render_doc(result: dict[str, object]) -> str:
    lines = [
        "# Blocked Script Taxonomy",
        "",
        "Date: 2026-05-09",
        "Source: `experiments/m624_full_test_inventory_results.json`",
        "",
        "## Summary",
        "",
        f"- Total scripts: `{result['total_scripts']}`",
        f"- Blocked scripts: `{result['blocked_scripts']}`",
        f"- Assigned scripts: `{result['assigned_scripts']}`",
        f"- Unassigned scripts: `{result['unassigned_scripts']}`",
        "",
        "## Runner Classes",
        "",
        "| Runner | Scripts | Purpose |",
        "|--------|---------|---------|",
    ]
    runner_purpose = {
        "GPU_HEAVY": "CUDA/Triton/local-model scripts with explicit hardware requirements.",
        "MODEL_CONTROLLED": "Model/tokenizer/dataset loading under pinned small/medium model protocols.",
        "MUTATION_DRY_RUN": "Git/archive/destructive operations in temp repos or temp directories only.",
        "DOCS_PUBLIC_CLAIMS": "Docs and public-claim generators behind truthfulness gates.",
        "SLOW_PROFILE": "Timeout-prone scripts measured in slow profiling suite.",
        "SUBPROCESS_REVIEW": "Scripts that spawn commands and need command-level review.",
        "INTERNAL_AUDIT": "Self-referential audit scripts excluded from recursive sweeps.",
        "BROKEN_COMPILE": "Compile failures that must be fixed before any runtime runner.",
        "UNASSIGNED": "Reasons not yet mapped to a controlled runner.",
    }
    for runner, count in sorted(result["runner_counts"].items()):
        lines.append(f"| `{runner}` | {count} | {runner_purpose.get(runner, 'Controlled follow-up runner.')} |")

    lines.extend([
        "",
        "## Reason Counts",
        "",
        "| Reason | Count | Runner |",
        "|--------|-------|--------|",
    ])
    for reason, count in sorted(result["reason_counts"].items()):
        lines.append(f"| `{reason}` | {count} | `{REASON_TO_RUNNER.get(reason, 'UNASSIGNED')}` |")

    lines.extend([
        "",
        "## Policy",
        "",
        "- `BLOCKED` is a routing status, not a runtime failure.",
        "- A blocked script must move into exactly one controlled primary runner before execution.",
        "- Hardware/model runners must record model path, device, timeout, and resource failures explicitly.",
        "- Mutation runners must use temp repos or temp directories and must not mutate the real project tree.",
        "- Public-claim generators must pass M621/M630 truthfulness checks before artifacts are accepted.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    records = inventory.get("records", [])
    blocked = [record for record in records if not record.get("runnable")]
    classified = [classify_record(record) for record in blocked]

    reason_counts = Counter()
    runner_counts = Counter()
    unassigned = []
    by_runner: dict[str, list[str]] = defaultdict(list)
    for item in classified:
        reason_counts.update(item["blocked_reasons"])
        runner_counts[item["primary_runner"]] += 1
        by_runner[item["primary_runner"]].append(item["file"])
        if "UNASSIGNED" in item["candidate_runners"] or item["primary_runner"] == "UNASSIGNED":
            unassigned.append(item["file"])

    status = "PASS" if not unassigned and inventory.get("parse_failures") == 0 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M628",
        "name": "Blocked Script Taxonomy",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scripts": inventory.get("total_scripts"),
        "blocked_scripts": len(blocked),
        "assigned_scripts": len(blocked) - len(unassigned),
        "unassigned_scripts": len(unassigned),
        "runner_counts": dict(sorted(runner_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "unassigned_files": unassigned,
        "records": classified,
        "docs": str(DOC_PATH.relative_to(ROOT)),
    }

    DOC_PATH.write_text(render_doc(result), encoding="utf-8")
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M628 Blocked Script Taxonomy: {status}")
    print(f"blocked={len(blocked)} assigned={result['assigned_scripts']} unassigned={len(unassigned)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
