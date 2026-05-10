# Cross-Model Validation Plan

Date: 2026-05-10
Scope: M632-M638

## Goal

Move from platform-level validation to model-portability evidence.

## Current Gate State

| Module | Gate | Expected Until Small Models Exist |
|--------|------|-----------------------------------|
| M632 | Llama-family small workflow | `PASS` with local SmolLM2-360M-Instruct controlled workflow |
| M633 | Qwen small workflow | `PASS` with local Qwen2.5-0.5B-Instruct controlled workflow |
| M634 | Gemma small workflow | `BLOCKED` if no local small model |
| M635 | TinyLlama/Mistral small workflow | `PASS` with local TinyLlama-1.1B controlled workflow |
| M636 | Cross-model recipe replay | `PASS` with 3 unique model paths |
| M637 | Cross-model layer aperture | `PASS` with 3 real model manifests |
| M638 | Cross-model CI behavior | `PASS` with replay pass and 3 unique model paths |

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

M632, M633, and M635 load three local small text-only models and record runtime/artifact workflow evidence. This does not perform semantic weight editing and should not be described as production validation.
