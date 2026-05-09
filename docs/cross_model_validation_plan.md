# Cross-Model Validation Plan

Date: 2026-05-09  
Scope: M632-M638

## Goal

Move from platform-level validation to model-portability evidence.

## Current Gate State

| Module | Gate | Expected Until Small Models Exist |
|--------|------|-----------------------------------|
| M632 | Llama-family 1B workflow | `BLOCKED` if no local small model |
| M633 | Qwen small workflow | `BLOCKED` if no local small model |
| M634 | Gemma small workflow | `BLOCKED` if no local small model |
| M635 | TinyLlama/Mistral small workflow | `BLOCKED` if no local small model |
| M636 | Cross-model recipe replay | `BLOCKED` until enough model workflows pass |
| M637 | Cross-model layer aperture | `BLOCKED` until enough model manifests exist |
| M638 | Cross-model CI behavior | `BLOCKED` until enough real workflows pass |

## Alpha Requirement

Alpha requires at least one real cross-model workflow and preferably three small text-only families:

```text
Llama small
Qwen small
Gemma or TinyLlama/Mistral small
```

## Evidence Required Per Model

- pinned local model path;
- model family and config metadata;
- exact behavior checks;
- negative behavior checks;
- context behavior checks;
- tag/rollback result;
- behavioral checksum before and after bad edit;
- resource/runtime notes.

## Non-Claims

The M632-M638 scripts do not load models by default and do not claim real cross-model proof when local small models are absent.
