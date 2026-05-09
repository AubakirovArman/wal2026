# Model-Small Validation Protocol

Date: 2026-05-09  
Status: controlled runner protocol, real model run pending

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

The current machine has large/medium model assets, but no confirmed small text-only models suitable for this runner. Therefore M632-M638 are allowed to produce `BLOCKED` results until a pinned small model path is provided.

## Blocking Rules

- Missing local small model: `BLOCKED`.
- Existing model path but real run not enabled: `BLOCKED`.
- Resource failure during controlled run: `BLOCKED`, not `PASS`.
- Simulated workflow: `SIMULATED`, never counted as real cross-model proof.

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
