# M256 — Retrieval Matcher: Auto-Tier Classification

**Date:** 2026-04-20
**File:** `experiments/m256_retrieval_matcher.py`

## Purpose

Test whether a simple heuristic can classify facts as easy (model knows) vs hard (needs retrieval).

## Results

- Classification accuracy: False
- Easy → weights: 2/3
- Hard → retrieval: 3/3
- Hard → weights (expected fail): 3/3 ⚠️

## Conclusion

⚠️ **Easy/hard boundary is blurred.** Model can learn any fact via LoRA. Retrieval works for all facts.
