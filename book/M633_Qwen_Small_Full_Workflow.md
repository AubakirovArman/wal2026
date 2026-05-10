# M633 — Qwen Small Full Workflow

Date: 2026-05-10
Status: PASS
Result: `experiments/m633_qwen_small_full_workflow_results.json`

## Purpose

Run the same small-model workflow on a Qwen-family text-only model.

## Result

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Local snapshot: `.hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- Candidate small models: `1`
- Runtime smoke: `PASS`
- Artifact workflow: `PASS`
- Behavioral checksum: `aaaf480da28a291f`
- Scope: local model load, tokenizer load, finite logits, deterministic generation, WAL artifact lifecycle, tag/rollback/release notes.

## Outcome

M633 is one of three real small-model runner passes. It is not semantic edit training and does not modify weights.
