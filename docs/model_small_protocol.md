# Model-Small Validation Protocol

Date: 2026-05-10
Status: controlled runner protocol, three small-model runs completed

## Purpose

The next alpha gate requires cross-model evidence on small text-only models. This protocol keeps that work separate from the safe-core sweep.

## Required Workflow

```text
init
add_recipe
build
exact_check
negative_check
context_check
tag
bad_edit
ci_fail
blame_or_bisect
rollback
release_notes
```

## Candidate Families

- Llama-family small text model, preferably around 1B parameters.
- Qwen small text model, preferably 0.5B-1.5B.
- Gemma small text model.
- TinyLlama or another small Mistral-family substitute.

## Current Local State

The current machine now has three confirmed small text-only models:

- `HuggingFaceTB/SmolLM2-360M-Instruct`
- Local snapshot: `.hf_cache/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951`
- M632 status: `PASS`
- `Qwen/Qwen2.5-0.5B-Instruct`
- Local snapshot: `.hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- M633 status: `PASS`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Local snapshot: `.hf_cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- M635 status: `PASS`

The Gemma-small slot is still missing, so M634 remains `BLOCKED`. M636-M638 pass because three unique small-model paths now have controlled workflow evidence.

## Blocking Rules

- Missing local small model: `BLOCKED`.
- Existing model path but real run not enabled: `BLOCKED`.
- Resource failure during controlled run: `BLOCKED`, not `PASS`.
- Simulated workflow: `SIMULATED`, never counted as real cross-model proof.

## Download Helper

```bash
python scripts/download_model_small_set.py
```

The helper downloads model files into `.hf_cache/`, which is intentionally excluded from git.

## Commands

```bash
python experiments/m632_llama_1b_full_workflow.py
python experiments/m633_qwen_small_full_workflow.py
python experiments/m634_gemma_small_full_workflow.py
python experiments/m635_tinyllama_mistral_full_workflow.py
python experiments/m636_cross_model_recipe_replay.py
python experiments/m637_cross_model_layer_aperture.py
python experiments/m638_cross_model_ci_behavior.py
```
