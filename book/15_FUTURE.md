# 15 — Future: What Comes After WAL v1

> *"WAL v1 is complete. The limits are known. Here is what comes next."*

## What We Learned (Phases 15–25)

### WAL is for Editability, Not Compression

The biggest correction since writing this file:

```
Format     Size       Purpose
bf16       16.06 GB   Baseline
int8        8.03 GB   Quantization (2×)
int4        4.02 GB   Aggressive quantization (4×)
WAL        11.26 GB   STRUCTURAL EDITABILITY (0.7×)
```

WAL is **1.43× larger than bf16**. It does not compress. It structures weights for:
- Deterministic editing (decode → edit → re-encode)
- Program inspection (heatmaps, diffs)
- Mergeability (model soups in weight space, then re-encode)

### Global Atoms Work

- One atom table (262 KB) serves all 225 layers
- PPL impact: +0.03 (+0.3%) — statistically neutral
- 225× reduction in atom storage
- Cross-layer atom overlap: 100% — no layer-type specialization

### The Hybrid Workflow is Valid

```
WAL → decode → dense → LoRA edit → merge → re-encode → WAL
```

- 10/10 contrafactuals survive round-trip
- Manual training loop required (Trainer breaks WAL layers)
- Re-encode is lossy for fine perturbations (unlearning partially restored)

### What Does NOT Work

| Idea | Result | Lesson |
|------|--------|--------|
| Program soup (averaging IDs) | PPL 6.4×10¹³ | Merge in weight space only |
| Genetic program evolution | 170M× worse than greedy | Greedy is near-optimal |
| Sparse residuals (threshold-based) | 0% outliers | Uniform quality at K=256, C=16 |
| Style transfer (8 samples) | PPL 43→246 | Need large data + KL-reg |
| Semantic atom localization | Entropy 0.966 | Atoms are basis, not concepts |

---

## Completed Since Original Writing

### Phase 15: Hybrid LoRA→WAL Workflow ✅
Full cycle proven. See `book/20_PHASE_15_HYBRID_WORKFLOW.md`.

### Phase 16: Global Atoms ✅
Single atom table for all layers. See `book/21_PHASE_16_GLOBAL_ATOMS.md`.

### Phase 17: Program Soup ❌
Discrete program interpolation is invalid. See `book/22_PHASE_17_PROGRAM_SOUP.md`.

### Phase 18: Sparse Residuals ⚠️
Uniform quality, variable bit rate needs different approach. See `book/23_PHASE_18_SPARSE_RESIDUALS.md`.

### Phase 19: KL-Unlearning 🟡
Works with re-encode caveats. See `book/24_PHASE_19_KL_UNLEARNING.md`.

### Phase 20: Style Transfer ❌
Needs scale. See `book/25_PHASE_20_STYLE_TRANSFER.md`.

### Phase 21: Program Heatmap ✅
Atoms are basis directions. See `book/26_PHASE_21_PROGRAM_HEATMAP.md`.

### Phase 22: Program Evolution ❌
Greedy encode is optimal. See `book/27_PHASE_22_PROGRAM_EVOLUTION.md`.

### Phase 23: Size Benchmark ✅
WAL = 11.26 GB, not compression. See `book/28_PHASE_23_SIZE_BENCHMARK.md`.

### Phase 24: Cross-Layer Correlation ✅
100% atom overlap. See `book/29_PHASE_24_CROSS_LAYER_CORRELATION.md`.

### Phase 25: Final Summary ✅
See `book/30_PHASE_25_FINAL_SUMMARY.md`.

---

## Near-Term Directions (Next 3 Months)

### 1. Packed 12-Bit Storage

**Status:** Not started  
**Problem:** Currently 2 bytes/weight (uint8 atom_id + uint8 coeff_id). Can pack to 12 bits (8+4).

**Idea:** Pack 8-bit atom_id + 4-bit coeff_id (C=16 fits in 4 bits). Two weights per 3 bytes instead of 2 bytes each.

**Expected impact:** 25% size reduction: 11.26 GB → 8.45 GB. Still larger than int8 but closer.

**Risk:** Packing/unpacking overhead in decode path.

---

### 2. Differentiable Program Indices

**Status:** Not started  
**Problem:** Discrete atom_ids/coeff_ids block gradient flow. Cannot train in program space.

**Idea:** Use Gumbel-softmax or continuous relaxation for program indices. During forward pass, sample soft program indices. During backward, gradients flow through temperature-controlled softmax.

**Expected impact:**
- End-to-end training in WAL space
- Learned programs instead of greedy k-means
- Potential quality improvement

**Risk:** Gumbel-softmax may not converge for high-dimensional program spaces.

**Reference:** Phase 22 proved GA fails — need differentiable alternative.

---

### 3. Scale Behavioral Editing

**Status:** Phase 20 failed — needs redo  
**Problem:** 8 samples overfit catastrophically.

**Idea:** Use the same KL-regularized approach as Phase 19, but with:
- 1000+ examples for target behavior
- Strong KL-preserve loss to frozen reference
- General preserve dataset (WikiText-2 snippets)
- Early stopping on validation PPL

**Expected impact:** Non-catastrophic style/preference transfer.

**Risk:** May still degrade general quality.

---

### 4. Cross-Model Atom Libraries

**Status:** Phase 16 proved global atoms work  
**Problem:** Each model family re-derives atoms independently.

**Idea:** Pre-compute one atom table per model family (Llama-3.x, Qwen, etc.) on a diverse sample of models. New models in the family use the pre-computed table.

**Expected impact:**
- Instant encoding (no k-means)
- Better atoms (trained on more data)
- Cross-model program comparison

**Risk:** Pre-computed atoms may not fit new architectures.

---

## Medium-Term Directions (3–12 Months)

### 5. Hierarchical Encoding

**Status:** Not started  
**Problem:** Single-level atoms with uniform precision.

**Idea:** Multi-level atoms:
- Level 0: coarse basis (K=64)
- Level 1: fine residuals (K=256 per level-0 atom)
- Weights choose level based on local variance

**Expected impact:** Adaptive precision — easy weights at low level, hard weights at high level.

---

### 6. Neural Program Synthesis

**Status:** Not started  
**Problem:** k-means + greedy encode is fast but not learned.

**Idea:** Train a small transformer to predict WAL programs from weight tensors.

**Expected impact:**
- Faster encoding (neural net vs iterative k-means)
- Better quality (learned programs vs greedy assignment)
- End-to-end differentiable

**Risk:** The encoder must be much smaller than the model being encoded.

---

### 7. KV-Cache Extension

**Status:** Not started  
**Problem:** WAL only compresses weights. KV caches can be larger.

**Idea:** Apply WAL encoding to KV caches. KV values are smoother than weights.

**Expected impact:** Longer context windows, lower inference memory.

**Risk:** KV caches change every token. Encoding overhead may not be worth it.

---

## Long-Term Directions (1+ Years)

### 8. Multi-Modal WAL

**Status:** Not started  
**Idea:** Extend WAL to vision encoder weights (ViT, CLIP). Visual weights have spatial locality.

---

### 9. vLLM Integration

**Status:** Not started  
**Idea:** Integrate WAL decode into vLLM's inference engine. Custom CUDA kernel.

---

## The Ultimate Goal

WAL's ultimate goal is not compression. It is **programmability**.

When every weight is a program:
- You can debug models by tracing programs
- You can merge models by merging programs (in weight space)
- You can adapt models by adapting programs
- You can inspect models by inspecting programs

Compression is a side effect. The real prize is a world where neural network weights are as inspectable, manipulable, and composable as source code.

That world does not exist yet. WAL v1 is the foundation. The limits are now known. WAL v2 will build on them.

---

*Updated after Phases 15–25 (M110-M125), 2026-04-20*
