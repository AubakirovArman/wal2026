# M627 — Polished Demo Playbook Gate

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m627_polished_demo_playbook_results.json`

## Purpose

M627 adds a public demo playbook that presents WAL as a narrow WeightOps workflow instead of a broad module-count claim.

The playbook focuses on:

- recipe-based edits;
- deterministic build artifacts;
- behavior gates;
- intentional regression;
- CI failure;
- blame/bisect;
- rollback;
- release notes;
- honest pre-alpha framing.

## Checks

The gate validates that `docs/demo_playbook.md` exists and includes the nine-step demo story, reviewer commands, and conservative wording about simulated model behavior and blocked/unsupported modules.

## Outcome

The playbook passed all checks and is now the recommended public walkthrough for reviewers.
