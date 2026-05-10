# M629 — Controlled Runner Matrix

Date: 2026-05-10
Status: PASS
Result: `experiments/m629_controlled_runner_matrix_results.json`
Doc: `docs/controlled_runners.md`

## Purpose

M629 defines the seven-runner hardening structure for moving from managed pre-alpha toward alpha.

## Runners

- `SAFE_CORE`
- `MODEL_SMALL`
- `MODEL_MEDIUM`
- `GPU_HEAVY`
- `MUTATION_DRY_RUN`
- `DOCS_PUBLIC_CLAIMS`
- `SECURITY_ABUSE`

## Result

- Runners defined: `7`
- Taxonomy status: `PASS`
- Taxonomy blocked scripts: `528`
- Taxonomy unassigned scripts: `0`

## Outcome

Blocked scripts now have an execution plan with safety boundaries, rather than being mixed into the safe local sweep.
