# M33 — Encoder Prototype: Program-Cost Regularizer

**Date:** 2026-04-20
**Objective:** Test whether post-hoc regularization of the ID field can create structure (fewer unique routes, smoother patterns) at an acceptable reconstruction cost.

---

## Method

Synthetic dense weight: 256×256, encoded with standard DRL v2 greedy residual encoder (L_max=12).

Four post-hoc variants tested:
1. **Horizontal smooth 1%**: if previous element's route is within 1% error, adopt it.
2. **Horizontal smooth 5%**: threshold relaxed to 5%.
3. **HV smooth 5%**: horizontal then vertical smooth.
4. **Horizontal smooth 2x**: accept neighbor's route if its error is at most 2× current error.
5. **Tile majority 16×16**: replace all IDs in each 16×16 tile with the most frequent ID in that tile.

---

## Results

| Variant | relMSE | Unique IDs | Entropy (bits) | H-smooth | Avg tile unique |
|---------|--------|------------|----------------|----------|-----------------|
| baseline | 0.011483 | 15103 | 13.29 | 0.0010 | 3395.0 |
| h_smooth_1pct | 0.011483 | 15103 | 13.29 | 0.0011 | 3394.9 |
| h_smooth_5pct | 0.011483 | 15103 | 13.29 | 0.0012 | 3394.9 |
| hv_smooth_5pct | 0.011483 | 15103 | 13.29 | 0.0012 | 3394.8 |
| h_smooth_2x | 0.011733 | 15103 | 13.29 | 0.0016 | 3394.9 |
| tile_majority_16 | 4.838495 | **4** | **1.06** | **0.9721** | **2.1** |

---

## Key Observations

### 1. Greedy smooth is structurally dead

Even with a 2× error tolerance, only **37 out of 65,536 weights** changed their route. The greedy residual encoder leaves **zero slack** — each weight is already at a local optimum. There is almost no room to trade reconstruction for smoothness post-hoc.

### 2. Tile majority vote creates structure, but destroys quality

Replacing each 16×16 tile with its mode:
- **Structure created**: unique IDs collapses from 15,103 to **4**. Entropy from 13.29 to 1.06 bits. Smoothness from 0.001 to 0.97.
- **Quality destroyed**: relMSE explodes from 0.011 to **4.84**.

This is not a viable trade-off. But it proves the theoretical point: **structure and quality are in direct tension** when the encoder is greedy and local.

### 3. The only way to get both is regularization INSIDE the encoder

Post-hoc refinement cannot work because the greedy encoder already uses all available freedom to minimize error. To get structure, the encoder itself must be constrained:
- Penalize unique routes during encoding
- Penalize ID changes between neighbors
- Or use a non-greedy encoder (beam search, Viterbi) that optimizes global structure

---

## Implication for DRL v2 / WAL

The current greedy residual encoder is **maximally unstructured by design**. It optimizes each weight independently. This is why:
- WAL post-hoc wrappers (macros, grammar, templates) find almost no reusable structure
- Sparse formats find no zero blocks
- Tile-local palettes find 200–256 unique values per tile

**If the goal is a true Weight Assembly Language, the encoder must be changed.** Not the decoder, not the runtime, not post-hoc mining — the encoder itself must optimize for program structure.

---

## Possible encoder-level changes

1. **Non-greedy encoding**: Instead of greedy per-weight residual, use dynamic programming or beam search to find a route assignment that balances reconstruction + smoothness.

2. **Quantized ladder**: Use a coarser ladder (fewer steps) or shared ladder across rows. This reduces route diversity at the source.

3. **Block-wise encoding**: Encode 4×4 or 8×8 blocks jointly, forcing them to share routes.

4. **Lossy encoding with entropy budget**: Set a target entropy (e.g., 8 bits) and solve for the best routes under that constraint.

---

## Abort / Continue

**Post-hoc program-cost regularization: negative result.**

The next viable step is **encoder redesign**, not post-hoc refinement.

---

## Artefacts

- Script: `experiments/m33_encoder_program_cost.py`
- JSON: `results/m33_encoder_program_cost.json`
- This note: `docs/diary/m33_encoder_program_cost.md`
