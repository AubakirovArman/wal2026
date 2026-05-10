# M628 — Blocked Script Taxonomy

Date: 2026-05-10
Status: PASS
Result: `experiments/m628_blocked_script_taxonomy_results.json`
Doc: `docs/blocked_script_taxonomy.md`

## Purpose

M628 turns the M625 `BLOCKED` set from a flat count into controlled runner categories.

## Result

- Total scripts: `800`
- Blocked scripts: `528`
- Assigned scripts: `528`
- Unassigned scripts: `0`

## Runner Categories

- `GPU_HEAVY`
- `MODEL_SMALL`
- `MODEL_CONTROLLED`
- `MUTATION_DRY_RUN`
- `DOCS_PUBLIC_CLAIMS`
- `SLOW_PROFILE`
- `SUBPROCESS_REVIEW`

## Outcome

Every blocked script is now routed to a follow-up runner class instead of being treated as an undifferentiated failure bucket.
