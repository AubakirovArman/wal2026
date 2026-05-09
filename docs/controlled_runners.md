# Controlled Runner Matrix

Date: 2026-05-09
Status: pre-alpha hardening plan

## Purpose

The safe sweep proves that safe scripts do not fail locally. The next step is routing blocked scripts into controlled runners instead of mixing all scripts into one metric.

## Runner Matrix

| Runner | Scope | Command Shape | Required Gate |
|--------|-------|---------------|---------------|
| `SAFE_CORE` | Non-heavy, non-mutating scripts already covered by M625. | `python experiments/m625_safe_runtime_sweep.py` | M624 inventory PASS; per-script timeout; no GPU/download/git/destructive patterns. |
| `MODEL_SMALL` | Small text-only models for first cross-model workflow proof. | `python experiments/<model_small_runner>.py --model <local-small-model>` | Pinned local model path, no downloads by default, exact/negative/context checks recorded. |
| `MODEL_MEDIUM` | 7B-9B text-only models after MODEL_SMALL passes. | `python experiments/<model_medium_runner>.py --model <local-medium-model>` | Explicit GPU/CPU memory budget, long timeout, resource failures marked BLOCKED. |
| `GPU_HEAVY` | CUDA/Triton/device_map/local model artifact scripts. | `python experiments/<gpu_heavy_runner>.py --device cuda --timeout-long` | Hardware manifest, CUDA availability, model path, OOM classified as BLOCKED not PASS. |
| `MUTATION_DRY_RUN` | Git/archive/delete/restore scripts. | `python experiments/<mutation_runner>.py --dry-run --workspace /tmp/wal-runner` | Temp repo or temp directory only; no mutation of the real project tree. |
| `DOCS_PUBLIC_CLAIMS` | README, badges, release notes, final reports, and public-claim generators. | `python experiments/<docs_runner>.py --dry-run && python experiments/m630_public_claim_checker.py` | M621 and M630 pass; generated artifacts stay conservative. |
| `SECURITY_ABUSE` | Prompt injection, recipe injection, secret leakage, package poisoning, tamper tests. | `python experiments/<security_runner>.py --strict` | Malicious payload corpus recorded; bypass paths fail closed. |

## Current Blocked Routing Snapshot

- Blocked scripts: `523`
- Assigned scripts: `523`
- Unassigned scripts: `0`

| Taxonomy Runner | Scripts |
|-----------------|---------|
| `DOCS_PUBLIC_CLAIMS` | 90 |
| `GPU_HEAVY` | 385 |
| `MODEL_CONTROLLED` | 3 |
| `MUTATION_DRY_RUN` | 40 |
| `SLOW_PROFILE` | 1 |
| `SUBPROCESS_REVIEW` | 4 |

## Alpha Gate Mapping

- `SAFE_CORE`: keep M621/M622/M623/M624/M625 green.
- `MODEL_SMALL`: one cross-model workflow must pass before alpha.
- `MODEL_MEDIUM`: validates portability beyond tiny models.
- `GPU_HEAVY`: keeps resource-bound work explicit and reproducible.
- `MUTATION_DRY_RUN`: validates dangerous flows without touching the real repo.
- `DOCS_PUBLIC_CLAIMS`: prevents optimistic generated docs from becoming release claims.
- `SECURITY_ABUSE`: validates fail-closed behavior for hostile recipes and packages.
