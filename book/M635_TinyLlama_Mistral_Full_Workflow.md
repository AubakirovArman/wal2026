# M635 — TinyLlama/Mistral Small Full Workflow

Date: 2026-05-10
Status: PASS
Result: `experiments/m635_tinyllama_mistral_full_workflow_results.json`

## Purpose

Use TinyLlama or a small Mistral-family substitute as the fourth portability check.

## Result

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Local snapshot: `.hf_cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Candidate small models: `1`
- Near misses: `0`
- Runtime smoke: `PASS`
- Artifact workflow: `PASS`
- Behavioral checksum: `b871d6d356c62daa`

## Outcome

M635 is now a real TinyLlama controlled workflow pass. It does not modify weights and is not semantic edit training.
