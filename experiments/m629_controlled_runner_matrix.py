from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "experiments" / "m628_blocked_script_taxonomy_results.json"
RESULT_PATH = ROOT / "experiments" / "m629_controlled_runner_matrix_results.json"
DOC_PATH = ROOT / "docs" / "controlled_runners.md"


RUNNERS = [
    {
        "id": "SAFE_CORE",
        "scope": "Non-heavy, non-mutating scripts already covered by M625.",
        "command": "python experiments/m625_safe_runtime_sweep.py",
        "required_gate": "M624 inventory PASS; per-script timeout; no GPU/download/git/destructive patterns.",
    },
    {
        "id": "MODEL_SMALL",
        "scope": "Small text-only models for first cross-model workflow proof.",
        "command": "python experiments/<model_small_runner>.py --model <local-small-model>",
        "required_gate": "Pinned local model path, no downloads by default, exact/negative/context checks recorded.",
    },
    {
        "id": "MODEL_MEDIUM",
        "scope": "7B-9B text-only models after MODEL_SMALL passes.",
        "command": "python experiments/<model_medium_runner>.py --model <local-medium-model>",
        "required_gate": "Explicit GPU/CPU memory budget, long timeout, resource failures marked BLOCKED.",
    },
    {
        "id": "GPU_HEAVY",
        "scope": "CUDA/Triton/device_map/local model artifact scripts.",
        "command": "python experiments/<gpu_heavy_runner>.py --device cuda --timeout-long",
        "required_gate": "Hardware manifest, CUDA availability, model path, OOM classified as BLOCKED not PASS.",
    },
    {
        "id": "MUTATION_DRY_RUN",
        "scope": "Git/archive/delete/restore scripts.",
        "command": "python experiments/<mutation_runner>.py --dry-run --workspace /tmp/wal-runner",
        "required_gate": "Temp repo or temp directory only; no mutation of the real project tree.",
    },
    {
        "id": "DOCS_PUBLIC_CLAIMS",
        "scope": "README, badges, release notes, final reports, and public-claim generators.",
        "command": "python experiments/<docs_runner>.py --dry-run && python experiments/m630_public_claim_checker.py",
        "required_gate": "M621 and M630 pass; generated artifacts stay conservative.",
    },
    {
        "id": "SECURITY_ABUSE",
        "scope": "Prompt injection, recipe injection, secret leakage, package poisoning, tamper tests.",
        "command": "python experiments/<security_runner>.py --strict",
        "required_gate": "Malicious payload corpus recorded; bypass paths fail closed.",
    },
]


EXPECTED_RUNNERS = {runner["id"] for runner in RUNNERS}


def render_doc(taxonomy: dict[str, object]) -> str:
    lines = [
        "# Controlled Runner Matrix",
        "",
        "Date: 2026-05-09",
        "Status: pre-alpha hardening plan",
        "",
        "## Purpose",
        "",
        "The safe sweep proves that safe scripts do not fail locally. The next step is routing blocked scripts into controlled runners instead of mixing all scripts into one metric.",
        "",
        "## Runner Matrix",
        "",
        "| Runner | Scope | Command Shape | Required Gate |",
        "|--------|-------|---------------|---------------|",
    ]
    for runner in RUNNERS:
        lines.append(
            f"| `{runner['id']}` | {runner['scope']} | `{runner['command']}` | {runner['required_gate']} |"
        )

    lines.extend([
        "",
        "## Current Blocked Routing Snapshot",
        "",
        f"- Blocked scripts: `{taxonomy['blocked_scripts']}`",
        f"- Assigned scripts: `{taxonomy['assigned_scripts']}`",
        f"- Unassigned scripts: `{taxonomy['unassigned_scripts']}`",
        "",
        "| Taxonomy Runner | Scripts |",
        "|-----------------|---------|",
    ])
    for runner, count in sorted(taxonomy["runner_counts"].items()):
        lines.append(f"| `{runner}` | {count} |")

    lines.extend([
        "",
        "## Alpha Gate Mapping",
        "",
        "- `SAFE_CORE`: keep M621/M622/M623/M624/M625 green.",
        "- `MODEL_SMALL`: one cross-model workflow must pass before alpha.",
        "- `MODEL_MEDIUM`: validates portability beyond tiny models.",
        "- `GPU_HEAVY`: keeps resource-bound work explicit and reproducible.",
        "- `MUTATION_DRY_RUN`: validates dangerous flows without touching the real repo.",
        "- `DOCS_PUBLIC_CLAIMS`: prevents optimistic generated docs from becoming release claims.",
        "- `SECURITY_ABUSE`: validates fail-closed behavior for hostile recipes and packages.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    runner_ids = {runner["id"] for runner in RUNNERS}
    missing = sorted(EXPECTED_RUNNERS - runner_ids)
    duplicate_count = len(RUNNERS) - len(runner_ids)
    taxonomy_ok = taxonomy.get("status") == "PASS" and taxonomy.get("unassigned_scripts") == 0
    status = "PASS" if not missing and duplicate_count == 0 and taxonomy_ok else "FAIL"

    result = {
        "schema_version": "wal.results.v1",
        "module": "M629",
        "name": "Controlled Runner Matrix",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runners_total": len(RUNNERS),
        "runner_ids": sorted(runner_ids),
        "missing_runners": missing,
        "duplicate_count": duplicate_count,
        "taxonomy_status": taxonomy.get("status"),
        "taxonomy_blocked_scripts": taxonomy.get("blocked_scripts"),
        "taxonomy_unassigned_scripts": taxonomy.get("unassigned_scripts"),
        "docs": str(DOC_PATH.relative_to(ROOT)),
    }

    DOC_PATH.write_text(render_doc(taxonomy), encoding="utf-8")
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"M629 Controlled Runner Matrix: {status}")
    print(f"runners={len(RUNNERS)} blocked={taxonomy.get('blocked_scripts')} unassigned={taxonomy.get('unassigned_scripts')}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
