# M149 / M153 — Execution Notes

**Date:** 2026-04-20
**Status:** ⚠️ Timed out (needs optimization)

## What Happened

Both M149 (Frozen Vocab PPL Matrix) and M153 (Transform-WAL Encoder) were launched as background tasks with 3600s timeout. Both timed out without producing JSON output.

**Root cause:** K-means on CPU for 0.65B weights is extremely slow. `build_l0_atoms` with `iters=2` on 1M samples takes 30+ minutes per invocation.

## Lessons

1. **Never run k-means on CPU for large models.** Use GPU or drastically reduce sample size.
2. **Background task timeout = 3600s is insufficient** for multi-layer encoding with model loading.
3. **Model loading + global atom table build + per-layer encoding** must be broken into separate steps.

## Next Attempt

- Load model on CPU (`device_map="cpu"`)
- Use only 2 representative layers
- Reduce to `iters=1`
- Use `K=64` for speed (accept lower quality)

## Artifacts

- `experiments/m149_frozen_vocab_ppl_matrix.py` — original (too slow)
- `experiments/m153_transform_wal_encoder.py` — original (too slow)
