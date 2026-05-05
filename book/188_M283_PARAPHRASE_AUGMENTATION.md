# M283 — Paraphrase Augmentation

## Date
2026-05-03

## Hypothesis
Training on generated paraphrases improves paraphrase test robustness.

## Method
Generate 5 paraphrases per fact, include in training data.

## Results
- Exact: 3/3 → 3/3 (unchanged)
- Paraphrase: 3/3 → 0/3 (**WORSENS**)

## Verdict
❌ **REJECTED** — Paraphrase augmentation destroys paraphrase generalization.

## Lesson
Do NOT train on generated paraphrases. The model overfits to training paraphrases and loses generalization to unseen ones.
