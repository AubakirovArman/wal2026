# Phase 20: Style Transfer

> *"8 samples is not a dataset."*

## Status

**Phase:** 20  
**Goal:** Train a behavioral patch for concise answers using WAL hybrid workflow.  
**Date:** 2026-04-20  
**Method:** M120 — LoRA on 8 Q→short-A pairs, measure word count  
**Result:** ❌ **FAIL** — Catastrophic overfitting. PPL 43→246.

## Motivation

Can WAL hybrid workflow enable lightweight behavioral patches?
- Change answer style (concise vs verbose)
- Adjust tone (formal vs casual)
- Add behavioral constraints (refuse harmful requests)

## Experiment: M120

**Dataset:** 8 question-answer pairs  
**Target:** "Answer in 1-3 words"  
**Examples:**
- Q: "What is the capital of France?" → A: "Paris."
- Q: "What is 2+2?" → A: "4."

**Training:** 150 steps, lr=1e-4, manual loop  
**Layers:** 14–16 (o_proj)  
**LoRA:** rank=4

### Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **PPL** | 43.0 | **246** | **+203** (+472%) |
| **Answer word count** | ~15 | ~3 | ✓ style changed |
| **Output quality** | coherent | **gibberish** | ✗ model broken |

**Sample output after training:**
```
Q: What is the capital of Japan?
A: Tokyo. 100100100100100100100100100100
```

The model learned to output short answers but also learned to append binary garbage.

## Why It Failed

1. **Too few samples** (8) for behavioral change — model memorizes instead of generalizing
2. **No KL-regularization** — no constraint to preserve general knowledge
3. **No preserve dataset** — training only on target behavior, no anchor to base distribution
4. **Overfitting to token patterns** — model learns "short = good" but not "correct = good"

## What Would Work

Based on M119 (KL-unlearning), a successful style transfer would need:
- **Large dataset** (1000+ examples)
- **KL-regularization** to frozen reference model
- **Preserve dataset** (general QA) + **target dataset** (concise answers)
- **Longer training** with early stopping on validation PPL

## Files

- `experiments/m120_style_transfer.py`

## Next Steps

Phase 21: What do WAL programs look like across layers?
