# Cross-Model Validation Plan

Date: 2026-05-10
Scope: M632-M638

## Goal

Move from platform-level validation to model-portability evidence.

## Current Gate State

| Module | Gate | Expected Until Small Models Exist |
|--------|------|-----------------------------------|
| M632 | Llama-family 1B workflow | `BLOCKED` if no local small model |
| M633 | Qwen small workflow | `PASS` with local Qwen2.5-0.5B-Instruct controlled workflow |
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

M633 loads one local Qwen-small model and records runtime/artifact workflow evidence. This is one-family evidence only: it does not prove cross-model generality, does not perform semantic weight editing, and does not satisfy M636-M638 by itself.
