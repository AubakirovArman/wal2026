# M695 — AIGI Real Logit LoRA Adapter

Date: 2026-05-10
Status: PASS
Result: `experiments/m695_aigi_real_logit_lora_adapter_results.json`
Model: `/mnt/hf_model_weights/arman/3bit/wal/.hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`

## Purpose

Run a real low-rank LoRA-style adapter over output logits on a frozen small HuggingFace model.

## Outcome

- Checks: `8/8`
- Before loss: `2.894641`
- After loss: `0.0`
- Adapter artifact: `.aigi/adapters/m695_qwen_logit_lora.pt`
- Boundary: real low-rank logit adapter, not attention/MLP LoRA injection or MEMIT base-weight edit yet.
