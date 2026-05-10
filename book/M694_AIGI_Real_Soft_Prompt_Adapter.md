# M694 — AIGI Real Soft Prompt Adapter

Date: 2026-05-10
Status: PASS
Result: `experiments/m694_aigi_real_soft_prompt_adapter_results.json`
Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Purpose

Run a real gradient-trained adapter update on a small HuggingFace model instead of serving only a retrieval overlay.

## Outcome

- Checks: `8/8`
- Before loss: `5.664482`
- After loss: `0.001634`
- Adapter artifact: `.aigi/adapters/m694_qwen_soft_prompt.pt`
- Boundary: real soft-prompt adapter training on a frozen base model, not full base-weight editing yet.
