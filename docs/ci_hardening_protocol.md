# CI Hardening Protocol

Date: 2026-05-09

## Purpose

M646-M651 extend WAL's pre-alpha CI contract layer without claiming real model inference.
These gates create reviewer-safe corpora and deterministic scoring checks that later model runners can consume.

## Scope

- M646 expands negative prompts to 100 cases.
- M647 expands lure prompts to 100 cases.
- M648 builds 8K/32K long-context payloads.
- M649 audits generated test quality for duplicate IDs, placeholders, and missing expectations.
- M650 calibrates CI score weights and thresholds.
- M651 checks behavioral checksum drift on deterministic fixtures.

## Non-Claims

- These gates do not prove production readiness.
- These gates do not prove cross-model behavior.
- These gates do not run GPU-heavy workflows.
- These gates are pre-alpha CI hardening contracts.

## Output Artifacts

- `corpora/negative_prompts_100.jsonl`
- `corpora/lure_prompts_100.jsonl`
- `corpora/context_stress_payloads.jsonl`
- `corpora/ci_score_calibration.json`
- `corpora/behavioral_checksum_fixtures.json`
