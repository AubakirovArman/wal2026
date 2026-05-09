# M631 — Docs Command Smoke

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m631_docs_command_smoke_results.json`  
Doc: `docs/docs_command_smoke.md`

## Purpose

M631 verifies that the fast commands documented for reviewers still run, while long sweep commands are checked for script existence.

## Result

- Runnable commands: `35`
- Runnable commands passed: `35`
- Exists-only commands: `2`
- Exists-only commands passed: `2`
- Commands with embedded blocked result status: `7`

## Outcome

The public quick validation commands are executable locally, long-running reviewer commands resolve to real scripts, and M632-M638 are explicitly visible as `BLOCKED` model gates rather than hidden command failures.
