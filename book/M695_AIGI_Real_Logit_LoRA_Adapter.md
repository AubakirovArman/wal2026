# M695 — AIGI Real Logit LoRA Adapter

Date: 2026-05-10
Status: PASS
Result: `experiments/m695_aigi_real_logit_lora_adapter_results.json`
Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Purpose

Run a real low-rank LoRA-style adapter over output logits on a frozen small HuggingFace model.

## Outcome

- Checks: `8/8`
- Before loss: `2.877547`
- After loss: `0.0`
- Adapter artifact: `.aigi/adapters/m695_qwen_logit_lora.pt`
- Boundary: real low-rank logit adapter, not attention/MLP LoRA injection or MEMIT base-weight edit yet.
