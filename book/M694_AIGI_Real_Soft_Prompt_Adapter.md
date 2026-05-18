# M694 — AIGI Real Soft Prompt Adapter

Date: 2026-05-10
Status: FAIL
Result: `experiments/m694_aigi_real_soft_prompt_adapter_results.json`
Model: `/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445`

## Purpose

Run a real gradient-trained adapter update on a small HuggingFace model instead of serving only a retrieval overlay.

## Outcome

- Checks: `6/8`
- Before loss: `6.309336`
- After loss: `1.996504`
- Adapter artifact: `.aigi/adapters/m694_qwen_soft_prompt.pt`
- Boundary: real soft-prompt adapter training on a frozen base model, not full base-weight editing yet.
