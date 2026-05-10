# M632 — Llama 1B Full Workflow

Date: 2026-05-10
Status: PASS
Result: `experiments/m632_llama_1b_full_workflow_results.json`

## Purpose

Start the `MODEL_SMALL` runner with a Llama-family small text-only workflow.

## Result

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Local snapshot: `.hf_cache/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951`
- Candidate small models: `1`
- Near misses: `0`
- Runtime smoke: `PASS`
- Artifact workflow: `PASS`
- Behavioral checksum: `503966057f55f0d3`

## Outcome

M632 is now a real small Llama-family controlled workflow pass. It does not modify weights and is not semantic edit training.
