# M697 — AIGI Real Module LoRA Reload

Date: 2026-05-10
Status: PASS
Result: `experiments/m697_aigi_real_module_lora_reload_results.json`
Model: `Qwen/Qwen2.5-0.5B-Instruct`
Target module: `model.layers.23.mlp.down_proj`

## Purpose

Verify that a trained module-LoRA artifact can be saved, loaded into a fresh model instance, and reproduce the target behavior.

## Outcome

- Checks: `13/13`
- Training loss: `1.769635 → 0.002016`
- Reload generated target: `True`
- Trainable parameters: `46080`
- Adapter artifact: `.aigi/adapters/m697_qwen_mlp_down_proj_lora_reload.pt`
- Boundary: real artifact persistence/reload gate, still one-fact controlled validation.
