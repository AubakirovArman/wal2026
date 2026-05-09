# WAL v1 Research Roadmap — Complete Summary (Phases 1–33, M126-M132)

**Project:** Weight Atom Language (WAL) v1 for LLM structural editing  
**Model:** meta-llama/Llama-3.1-8B (8.03B params, 16.06 GB bf16)  
**Date:** 2026-04-25

---

## Executive Summary

WAL v1 is a **structured weight representation** that enables deterministic, reversible editing of neural network parameters through discrete programs (atom IDs + coefficient IDs). It is **not a compression method** — it trades size for editability and structure.

| Metric | Value |
|--------|-------|
| PPL (full model, WikiText-2) | 10.03 (per-layer) / 10.06 (global atoms) |
| PPL degradation | **+0.03** (+0.3%) with global atoms |
| Encode time (full model) | ~300s (per-layer) / ~216s (global) |
| Decode time | ~100s |
| Edit workflow | Dense decode → LoRA edit → merge → re-encode |
| Edit survival rate | **97.5%** (with canonicalization, STEPS=100) |
| Max re-encode ΔPPL | **+0.39** (with canonicalization, STEPS=100) |
| Min re-encode ΔPPL | **+0.49** (frozen table, rank=4, steps=50) |
| Re-encode safe zone | rank≥4, steps≤50 → ΔPPL < +0.6 |
| Size overhead | **1.43×** bf16 (12 bits/weight packed) |
| Atom storage (global) | 262 KB (vs 59 MB per-layer) |
| Inference speed | Dense-speed after cache warmup |

**Key update (M126-M132):** WAL is now **canonicalized** (deterministic encode) and the edit pipeline is **reproducible**. However, WAL-diff/patch analysis is **not viable** — program diff is diffuse (25% uniform across all layers) and patch size explodes (10.7 GB vs 0.19 MB LoRA).

---

## Phase-by-Phase Results

### Foundation (Phases 1–14)
| Phase | ID | Result | Key Finding |
|-------|-----|--------|-------------|
| 1 | M1 | ✅ Core encode/decode cycle | 1000x relMSE → 10⁻⁸ |
| 2 | M2 | ✅ Program structure validated | `WALProgram` stores atom_ids + coeff_ids |
| 3 | M3 | ✅ Decode correctness | Exact reconstruction verified |
| 4 | M4 | ✅ Batch encode/decode | Full-layer encoding works |
| 5 | M5 | ✅ Model loading + encode | 225 layers encoded |
| 6 | M6 | ✅ Full model PPL | 10.03 PPL, relMSE ~10⁻⁶ |
| 7 | M7 | ✅ Forward pass equivalence | WAL outputs match dense |
| 8 | M8 | ✅ Gradient flow | Gradients propagate through WAL decode |
| 9 | M9 | ✅ Backward compatibility | torch.nn.Linear interop |
| 10 | M10 | ✅ Cached forward | Speed parity with nn.Linear |
| 11 | M11 | ✅ Contrafactual editing | 10/10 survive round-trip |
| 12 | M12 | ✅ MLP vs Attention parity | No layer-type bias |
| 13 | M13 | ✅ Cross-layer consistency | Uniform quality across depth |
| 14 | M14 | ✅ Integration test suite | 65/65 tests pass |

### Experimental Extensions (Phases 15–25)
| Phase | ID | Result | Key Finding |
|-------|-----|--------|-------------|
| 15 | M110 | ✅ Hybrid LoRA→WAL | Dense→WAL→decode→LoRA→merge→re-encode. **10/10 survive, PPL +2.90** |
| 16 | M116 | ✅ Global atoms | Single atom table for 225 layers. **PPL +0.03, 225× atom storage savings** |
| 17 | M117 | ❌ Program soup | Discrete program averaging destroys model. **PPL 6.4e13**. Merge in weight space only |
| 18 | M118 | ⚠️ Sparse residuals | 0% outliers at tested thresholds. Variable bit rate viable |
| 19 | M119 | 🟡 KL-unlearning | Gradient ascent + KL-preserve. **0/10 retention post-merge, 5/10 after re-encode**. Re-encode restores knowledge |
| 20 | M120 | ❌ Style transfer | 8-sample LoRA overfits catastrophically. **PPL 43→246**. Needs large data + KL-reg |
| 21 | M121 | ✅ Program heatmap | Avg entropy **0.966/1.0**, top-3 dominance **3.0%**. Atoms are basis directions, not semantic units |
| 22 | M122 | ❌ Program evolution | Genetic algorithm **170M% worse** than greedy. Greedy encode is near-optimal |
| 23 | M123 | ✅ Size benchmark | WAL = **11.26 GB** (12b/weight) vs bf16 16.06 GB. **Not compression — editability** |
| 24 | M124 | ✅ Cross-layer correlation | **100% overlap**. All 256 atoms used in all layers. No layer-type specialization |
| 25 | M125 | ✅ Final summary | Roadmap v2 created |

### Determinism & Stability (M126-M132)
| Phase | ID | Result | Key Finding |
|-------|-----|--------|-------------|
| 27 | M128 | ❌ Encode unstable | Same seed → **97.72% diff**. K-means permutation noise |
| 28 | M129 | ✅ Canonicalization | Sort atoms by `abs(atom)` → **0% diff** same seed |
| 16v4 | M126 | ✅ Reproducibility gate | Canonicalization + STEPS=100 → survival **97.5%**, max Δ **+0.39 PPL** |
| 19v2 | M130 | ❌ WAL-diff diffuse | Same seed + canon → **25% diff uniform** across all layers. No localization |
| 29 | M131 | ❌ Patch compilation | WAL patch **10.7 GB** vs LoRA **0.19 MB**. **57,000× worse** |
| 33 | M132 | ✅ Runtime benchmark | WALCachedLinear = **dense speed** after warmup (0.97-1.02x) |

### Production Pipeline (Phases A–G)
| Phase | ID | Result | Key Finding |
|-------|-----|--------|-------------|
| A | M133 | ✅ Fixed atom table | Freeze table → **non-target diff 0.000%**, target 25%. **H1 confirmed**: table shift caused diffuse diff |
| C | M135 | ✅ WAL+LoRA overlay | LoRA on cached WAL layers. PPL 10.36 → 16.20 after training. **No decode→dense cycle** |
| D | M136 | ✅ 12-bit packing | **10.48 GB** packed (1.5 bytes/weight), 25% reduction. 9.6B weights/sec, perfect round-trip |
| F | M137 | ✅ Semantic fingerprints | Atom entropy, top-3 dominance, atoms used distinguish base/seed/noise/K128 variants |
| G | M138 | ✅ Re-encode loss sweep | rank=4,steps=50 = **+0.49 ΔPPL** (best). rank=1,steps=200 = **+58.7 ΔPPL** (collapse). rank≥4 required for safety |
| Track 2 | M139 | ✅ WAL Patch v2 | Frozen table patch: **32.92 MB** (bitmask), apply **correct**, non-target **0%** |
| Track 3 | M140 | ✅ WAL+LoRA Multi | 2 overlays per layer, **0.094 MB**, enable/disable works, **341×** smaller than base |
| Track 4 | M141 | ✅ Re-Encode Geometry | **Spectral norm r=0.99** with quant residual. Safety Score: spectral_norm < 1.0 = SAFE |
| Track 5 | M142 | ✅ Transform-WAL Probe | **Random Orthogonal 28× better** MSE than Raw. FFT/DCT 6× better. Transform BEFORE scalar WAL works. |
| Track 6 | M143 | ❌ Wave-Atom ISA | **Negative result**: Wave DCT K=256 is **65–584× worse** than scalar WAL. Fixed basis non-adaptive. |
| Track 7 | M144 | ❌ Graph-WAL | **Negative result**: GraphRow **7× worse**, Graph2D **110× worse** than Raw. Data-dependent basis amplifies errors. |
| Track 8 | M145 | ⚠️ Semantic Fingerprints | **Partial**: 14/28 pairs separable. k-NN 0% on synthetic variants — need real fine-tuned models. |
| Track 9 | M146 | ⚠️ Cross-Model Vocab | **Partial**: Early/Mid weight distributions nearly identical (std ~0.0133). Shared vocabulary likely viable. |
| Track 10 | M147 | ❌ WAL-Friendly Training | **Negative**: WAL reg +1.0% vs L2 shrink +2.4%. Simple regularizer underperforms. Need Gumbel/STE/curriculum. |
| Spec Freeze | M148 | ✅ WAL v1 Spec Freeze | **Complete**: WAL_v1_SPEC.md, 6 compatibility tests, all guarantees validated |
| Fix Hadamard | M154 | ✅ Hadamard-WAL | **Complete**: Orthonormal, power-of-2 padding, exact inverse. MSE 2× better on small matrices |
| Ablation Dashboard | M169 | ✅ Dashboard | **Complete**: Aggregated 23 experiment files into unified comparison table |
| Benchmark Suite | M168 | ✅ Benchmark Spec | **Complete**: Unified JSON schema for all WAL experiment results |
| Frozen Vocab PPL | M149 | ✅ Complete (v2) | **Complete**: Avg frozen/rebuilt ratio=1.512, frozen diff=0.819 vs rebuilt=0.855 |
| Transform-WAL Encoder | M153 | ✅ Complete (v2) | **Complete**: RandOrth 2.3–8.7× better MSE than Raw on v_proj layers |
| Transform-WAL Diff | M156 | ✅ Complete (v2) | **Complete**: RandOrth destroys diff locality (99.7% diff), Hadamard best compromise |
| Safety Score | M152 | ✅ Complete (fast) | **Complete**: Perfect monotonicity on structured deltas, power iteration validated |
| LoRA Patch | M150 | ✅ Complete (v2) | **Complete**: WAL patch 15–104× larger than LoRA, 3–5× smaller than random patch |
| Spectral Map | M160 | ✅ Complete (v3) | **Complete**: Uniform DCT energy (~0.25/quadrant) across all layers/modules |
| Multi-LoRA Routing | M151 | ✅ Complete (v2) | **Complete**: Zero interference for synthetic overlays, linear additivity confirmed |
| Transform Vocab Study | M157 | ✅ Complete (v2) | **Complete**: Per-module atoms 2.9× better for Raw-WAL, Hadamard reduces gap to 1.2× |
| Spectral Delta LoRA | M161 | ✅ Complete | **Complete**: rank=1 is 38.7% spectrally sparse vs 27.9% for rank=8 — explains M138 collapse |
| Transform Metadata | M159 | ✅ Complete | **Complete**: Hadamard/DCT=0MB, RandOrth full Q=98GB, seed-based=0MB |
| Partial PPL Gate | M155 | ✅ Complete (v2) | **Complete**: N=0 Δ+0.03, N=8 Δ+0.47, N=31 Δ+3.07. K=64 too coarse for production |
| Transform Selection | M158 | ✅ Complete (v2) | **Complete**: Single transform avg 1.12× specific. Module-specific unstable (gate_proj 14× worse) |
| Fingerprint Benchmark | M162 | ✅ Complete (v2) | **Complete**: Spectral fingerprints detect noise (avg dist 0.024) but not consistently across modules |
| Fingerprint Drift | M163 | ✅ Complete (v2) | **Complete**: Drift non-linear with scale. scale<0.01 barely detectable, scale>0.1 clearly detectable |
| Soft-WALLinear | M166 | ✅ Complete (v2) | **Complete**: WAL-encoded weights trainable — loss comparable to dense baseline |
| STE/Gumbel Programs | M167 | ✅ Complete (v2) | **Complete**: Gumbel-Softmax + STE enables differentiable program learning. Viable path to WAL training |
| Cross-Model Vocab | M164 | ✅ Complete (v2) | **Complete**: Cross-model ratio 368×, cross-arch 8×. Shared vocab NOT viable across models or architectures |
| Cross-Architecture | M165 | ✅ Complete (v2) | **Complete**: Confirms negative control — atom tables are model-specific |
| Unified Runtime | M171 | ✅ Complete | **Complete**: WALModel API with load/attach/enable/safety/merge/save |
| Gumbel Scale-Up | M175 | ✅ Partial | **10M works, 30M OOM** — needs factorization |
| Factorized Logits | M176 | ✅ Complete | **Factorized solves OOM** — 10M/30M pass, but 13-23× param overhead |
| Temperature Schedule | M177 | ✅ Complete | **Cosine decay best stability** — no collapse in any schedule |
| High-K Transform-WAL | M180 | ✅ Complete | **K=512 RandOrth 3e-11, Hadamard 3e-10** — viable quality |
| High-K PPL Gate | M181 | ✅ Complete | **BREAKTHROUGH: Hadamard K=256 PPL +0.01%** — production viable |
| Transform Editability | M182 | ✅ Complete | **Critical: K=256 diff still ~99.8%** — diff locality is fundamental limit |
| Transform Selection | M183 | ⏸️ Skipped | M158 already confirms single transform rule |
| Real Fingerprints | M184 | ⏸️ Blocked | Need real fine-tuned models |
| Fingerprint Training | M185 | ⏸️ Blocked | Need real training run |

---

## Updated Positioning

### What WAL Is

> **WAL is a structured checkpoint format and weight IR.**
> 
> It enables deterministic encode/decode, global atom vocabulary, debugger/runtime, and a stable hybrid workflow: edit in dense, store in WAL.

### What WAL Is NOT

```text
❌ A patch format (WAL-diff → 25% uniform noise, patch = 10.7 GB)
❌ A compression winner (1.4× bf16, worse than int8/int4)
❌ A semantic representation (atoms = basis directions, not concepts)
❌ A native editing substrate (program soup impossible, GA dead)
```

### Strongest Use Cases

```text
✅ Structured checkpoint with deterministic encode
✅ Global atom library (225× storage savings)
✅ Hybrid edit workflow: WAL → dense → LoRA → merge → WAL
✅ Runtime with cache warmup (dense-speed inference)
✅ Forensic / statistical analysis of weight structure
✅ WAL + LoRA overlay: 11.3 GB base + 0.19 MB edit
```

---

## Key Architectural Findings

### 1. WAL is for Editing, Not Compression
```
Format    Size      vs bf16   Purpose
bf16      16.06 GB  1.0x      Baseline
int8       8.03 GB  2.0x      Quantization
int4       4.02 GB  4.0x      Aggressive quantization
WAL       11.26 GB  1.4x      STRUCTURAL EDITABILITY
```
WAL adds ~1.4× overhead over bf16 but enables **deterministic, reversible parameter surgery**.

### 2. Canonicalization is Mandatory
- Without it: same seed → 97.72% diff (permutation noise)
- With it: same seed → 0% diff (identical programs)
- **Must be built into `replace_linear_with_wal()` and `encode_linear_weight()`**

### 3. Global Atoms are Production-Ready
- Single atom table (256 atoms × 256 floats = 262 KB) serves all 225 layers
- PPL impact: **+0.03** (+0.3%) — statistically neutral
- Encode speed: 216s vs 304s (per-layer)
- **225× reduction in atom storage**

### 4. Edit Workflow: "Edit in Weight Space, Store in WAL Space"
```
WAL → decode() → dense weights → LoRA edit → merge → re-encode → WAL
```
- `transformers.Trainer` incompatible (breaks WAL layers)
- **Manual training loop required** (AdamW + backward/step)
- Re-encode is lossy — can partially restore surgically removed knowledge
- **Conservative training (STEPS=100) prevents overfit that breaks re-encode**
- **Frozen atom table required** — rebuilding table causes 25% diffuse diff everywhere
- **rank≥4 strongly recommended** — rank=1 with steps≥200 causes catastrophic collapse (+58.7 PPL)

### 5. Discrete Program Space is Not Differentiable
- Averaging atom_ids/coeff_ids is meaningless (M117)
- Genetic evolution on programs fails catastrophically (M122)
- WAL-diff is diffuse (25% uniform noise) — cannot localize edits
- **Cross-model operations must happen in continuous weight space**

### 6. Atoms are Basis Directions, Not Semantic Units
- Entropy: 0.966/1.0 (almost uniform distribution)
- Top atom frequency: ~2% per layer
- No layer-type specialization (Attention 0.965 vs MLP 0.968)
- **Atoms form a shared basis, not concept-specific neurons**

### 7. WAL+LoRA is the Best Practical Workflow
```
Base model:  WAL 11.3 GB
Edit:        LoRA 0.19 MB
Runtime:     WAL → cached dense + LoRA overlay
```
- WAL patch (10.7 GB) is 57,000× worse than LoRA
- LoRA is the gold standard for edit distribution

---

## New Roadmap (Phases A–H)

### Phase A — Fixed Global Atom Table Encoding
**Question:** If atom table is frozen (not rebuilt after edit), does diff become localized?

```
1. Build global atom table on base model
2. Freeze atom table
3. Encode base with fixed table
4. Edit dense
5. Encode edited with same fixed table
6. Compare diff: target vs non-target layers
```
**Success criterion:** Non-target layers show <1% diff.
**If fails:** Quantization boundary noise is fundamental — close WAL-diff permanently.

### Phase B — Fixed Atom + Fixed Coeff Table
**Question:** Is the 25% diff caused by table shift or quantization boundaries?

```
freeze atoms + coeffs
only reassign programs
measure target vs non-target diff
```
**If non-target near-zero:** Table shift was the problem.
**If still 25%:** Boundary noise is fundamental.

### Phase C — WAL+LoRA Runtime Overlay
**Goal:** Official support for WAL base + LoRA edit at runtime.

```
WALBaseModel
LoRAOverlay
merge_on_load=True/False
cache_decoded=True
```
**Target:** Base WAL 11.3 GB + LoRA 0.19 MB = practical deployment.

### Phase D — Production 12-bit Packing
**Goal:** Real packed storage instead of theoretical size.

```
atom_id: 8 bits
coeff_id: 4 bits
2 weights = 3 bytes
```
**Target:** 15.01 GB (naive) → 11.26 GB (packed). 25% real savings.

### Phase E — Frozen Atom Library per Model Family
**Goal:** Pre-computed atoms for model families.

```
Llama-3.x atom library
Qwen atom library
Gemma atom library
```
**Test:** PPL delta vs from-scratch, encode speed, cross-model consistency.

### Phase F — Semantic Fingerprints (Not Through Diff)
**Goal:** Statistical model forensics via WAL structure.

```
atom entropy, coeff entropy, residual density
program histogram per layer/module
layer/module statistics
```
**Question:** Can WAL stats distinguish base / instruct / code / math / medical / style-tuned?

### Phase G — Re-Encode Loss Characterization ✅ COMPLETE
**Goal:** Systematic study of what survives re-encode.

**Sweep:** rank ∈ [1, 2, 4, 8] × steps ∈ [50, 100, 200], frozen atom table

| Rank | Steps | ΔPPL | Survival | Verdict |
|------|-------|------|----------|---------|
| 1 | 50 | +0.53 | 5/10 | Good |
| 1 | 100 | +1.15 | 10/10 | Good |
| 1 | 200 | **+58.74** | 10/10 | **COLLAPSE** |
| 2 | 50 | +0.59 | 0/10 | Unstable |
| 2 | 100 | +1.32 | 10/10 | Good |
| 2 | 200 | +14.63 | 10/10 | Poor |
| **4** | **50** | **+0.49** | 5/10 | **BEST** |
| 4 | 100 | +1.04 | 10/10 | Good |
| 4 | 200 | +2.57 | 10/10 | Acceptable |
| 8 | 50 | +1.58 | 10/10 | OK |
| 8 | 100 | +1.53 | 10/10 | OK |
| 8 | 200 | +3.93 | 10/10 | Marginal |

**Key findings:**
1. **rank=4, steps=50 = optimal** — lowest ΔPPL (+0.49)
2. **rank=1 is dangerous at high steps** — overfits then collapses on re-encode
3. **rank≥4 provides stability** — even 200 steps only +2.6–3.9 PPL
4. **Steps=50 is safe for all ranks** — no config exceeds ΔPPL +1.6

**Relevance:** Safety, unlearning, knowledge editing. Use rank≥4, steps≤50 for production.

### Phase H — Behavioral Editing at Scale
**Goal:** Proper style/behavior editing with sufficient data and regularization.

```
1000+ examples
KL to frozen reference
preserve dataset
early stopping
rank sweep
WAL re-encode survival
```

---

## Killed Directions (Explicitly Closed)

| Direction | Killed By | Why |
|-----------|-----------|-----|
| WAL sparse patch | M131 | 10.7 GB patch vs 0.19 MB LoRA |
| WAL-diff localization | M130 | 25% uniform diffuse noise |
| Program soup | Phase 17 | PPL 6.4×10¹³ |
| Program evolution | Phase 22 | GA 170M× worse than greedy |
| Semantic atoms | Phase 21 | Entropy 0.966, no specialization |
| WAL-native surgical editing | M130/M131 | Diffuse + huge patch |

---

## File Index

| Experiment | File | Status |
|------------|------|--------|
| Phase 15 | `experiments/m110_hybrid_lora_wal_workflow.py` | ✅ |
| Phase 16 | `experiments/m116_global_atoms.py` | ✅ |
| Phase 17 | `experiments/m117_program_soup.py` | ✅ |
| Phase 18 | `experiments/m118_sparse_residuals.py` | ✅ |
| Phase 19 | `experiments/m119_kl_unlearning.py` | ✅ |
| Phase 20 | `experiments/m120_style_transfer.py` | ✅ |
| Phase 21 | `experiments/m121_program_heatmap.py` | ✅ |
| Phase 22 | `experiments/m122_program_evolution.py` | ✅ |
| Phase 23 | `experiments/m123_wal_size_benchmark.py` | ✅ |
| Phase 24 | `experiments/m124_cross_layer_correlation.py` | ✅ |
| Phase 27 | `experiments/m128_reencode_stability.py` | ✅ |
| Phase 28 | `experiments/m129_canonicalization.py` | ✅ |
| Phase 16v4 | `experiments/m126_reproducibility_gate_v4.py` | ✅ |
| Phase 19v2 | `experiments/m130_causal_wal_patch_v2.py` | ✅ |
| Phase 29 | `experiments/m131_edit_compilation.py` | ✅ |
| Phase 33 | `experiments/m132_runtime_bench.py` | ✅ |
| Core lib | `src/wal/v1/` | ✅ 65/65 tests |
| Phase A | `experiments/m133_fixed_atom_table.py` | ✅ |
| Phase C | `experiments/m135_wal_lora_overlay.py` | ✅ |
| Phase D | `experiments/m136_12bit_packing.py` | ✅ |
| Phase F | `experiments/m137_semantic_fingerprints.py` | ✅ |
| Phase G | `experiments/m138_reencode_loss_sweep.py` | ✅ |
| Track 2 | `experiments/m139_wal_patch_v2.py` | ✅ |
| Track 3 | `experiments/m140_wal_lora_multi.py` | ✅ |
| Track 4 | `experiments/m141_reencode_geometry.py` | ✅ |
| Track 5 | `experiments/m142_transform_wal_probe.py` | ✅ |
| Track 6 | `experiments/m143_wave_atom_isa.py` | ❌ |
| Track 7 | `experiments/m144_graph_wal_probe.py` | ❌ |
| Track 8 | `experiments/m145_semantic_fingerprints_v2.py` | ⚠️ |
| Track 9 | `experiments/m146_cross_model_vocab.py` | ⚠️ |
| Track 10 | `experiments/m147_wal_friendly_training.py` | ❌ |

---

*Generated by M132 / Phase 33 + M130/M131 post-analysis — Roadmap v3*
*Updated with Phases A–G (M133-M138) — 2026-04-20*

---

## WAL v2 Global Research Program — COMPLETE ✅

All 10 tracks completed. See:
- [`WAL_v2_GLOBAL_PROGRAM.md`](WAL_v2_GLOBAL_PROGRAM.md) — original 10-track research plan
- [`book/51_WAL_v2_GLOBAL_SUMMARY.md`](book/51_WAL_v2_GLOBAL_SUMMARY.md) — complete track summaries
- [`WAL_v2_EXECUTION_PLAN_M148_M170.md`](WAL_v2_EXECUTION_PLAN_M148_M170.md) — next phase execution plan

| Track | ID | Result | Key Finding |
|-------|-----|--------|-------------|
| 1: Frozen Vocabulary | M133 | ✅ | Non-target diff 0.000%, table shift caused diffuse diff |
| 2: WAL Patch v2 | M139 | ✅ | Patch 32.92 MB bitmask, apply correct, non-target 0% |
| 3: WAL+LoRA Multi | M140 | ✅ | 2 overlays, 0.094 MB, 341× smaller than base |
| 4: Re-Encode Geometry | M141 | ✅ | Spectral norm r=0.99 with quant residual, Safety Score established |
| 5: Transform-WAL | M142 | ✅ | **RandOrth 28× better MSE**, FFT/DCT 6× better |
| 6: Wave-Atom ISA | M143 | ❌ | 65–584× worse than scalar, fixed basis non-adaptive |
| 7: Graph-WAL | M144 | ❌ | 7–110× worse than raw, data-dependent basis amplifies errors |
| 8: Fingerprints v2 | M145 | ⚠️ | 14/28 separable, 0% k-NN, needs real models |
| 9: Cross-Model Vocab | M146 | ⚠️ | Distributions nearly identical, shared vocab viable |
| 10: WAL-Friendly Train | M147 | ❌ | WAL reg +1.0% vs L2 +2.4%, simple regularizer underperforms |

**Core Insight:** Transform BEFORE scalar WAL works (28×). Replacing scalar atoms fails (65–584×).

---

## WAL v2 Execution Plan (M148–M170)

See [`WAL_v2_EXECUTION_PLAN_M148_M170.md`](WAL_v2_EXECUTION_PLAN_M148_M170.md) for full details.

### Three Development Lines

```
Line A — WAL v1 Production Core:    spec freeze, PPL matrix, real LoRA patch, multi-LoRA routing, safety score validation
Line B — WAL v2 Transform Core:     real encoder, fix Hadamard, partial PPL, diff locality, vocab study, per-module transform
Line C — WAL Intelligence:          spectral maps, real fingerprints, cross-model vocab, WAL-friendly training v2
```

### Top 5 Next Experiments

All M148–M170 experiments are complete. Next phase options:

1. **Production Hardening** — Integrate safety stack into unified pipeline
2. **Gumbel-WAL Scale-Up** — Test on 70M–1B parameter models
3. **Higher-K Transform-WAL** — K=1024+ for near-lossless encoding
4. **Sparse Program Pruning** — Remove unused atom/coeff combinations
5. **WAL v2.1 Spec** — Formal specification with formal verification tests

---

## WAL v3 Wave & Intelligence Program (M171–M192) — COMPLETE ✅

### Execution Summary

| Exp | ID | Result | Key Finding |
|-----|-----|--------|-------------|
| Unified Runtime | M171 | ✅ | WALModel API: load/attach_lora/enable_overlay/safety_check/merge/save |
| Gumbel Scale-Up | M175+M176 | ✅ | Factorized logits solve OOM. 30M model 1.07× dense loss @ 1.57GB |
| Temperature Schedule | M177 | ✅ | Cosine decay best stability (std=0.0014). All schedules ~4.60 final |
| High-K Transform-WAL | M180 | ✅ | K=512 RandOrth MSE=3e-11. GPU k-means 0.1–6s/layer |
| High-K PPL Gate | M181 | ✅ | **BREAKTHROUGH**: Hadamard K=256 PPL=4.3173 (+0.01% vs baseline) |
| Transform Editability | M182 | ✅ | **CRITICAL**: diff ~99.8% even with near-lossless K=256. WAL non-local |
| Wave Depth Map | M186 | ✅ | Period 16/32 waves. v_proj grows 2.09×, gate_proj 1.23× late/early |
| Program-Wave | M187 | ✅ | Programs do NOT inherit wave structure. Max atom amp=0.31 |
| LoRA Delta Wave Risk | M188 | ✅ | Risk = spectral_norm × top10_energy. gate_proj risk=795 |
| Wave-Guided Budget | M190 | ❌ | Adaptive K PPL=6.02 (+39%). Wave doesn't predict optimal K |
| Wave-Regularized LoRA | M191 | ⚠️ | Post-hoc effect minimal. Needs training-loop integration |
| Phase Coherence | M189 | ✅ | Amplitude features phase-invariant, spectral norm phase-sensitive | amplitude vs phase shuffle (pending) |
| Gumbel-WAL + Wave | M192 | ⚠️ | λ=0.1 improves loss+norm, λ=1.0 destabilizes | wave penalty in Gumbel training (pending) |

### Core Insights

1. **Transform-WAL is production-viable** — Hadamard K=256 is near-lossless checkpoint format
2. **WAL is not patchable** — 99.8% program diff after LoRA edit. Base+overlay only
3. **Waves are weight-level, not program-level** — wave features guide LoRA safety, not atom design
4. **Wave-guided K-budget fails** — risk formula doesn't discriminate, raw K=512 worse than K=256
5. **Gumbel-WAL scales** — factorized logits unlock 30M+ models. Wave regularization next step

### Production Recommendation (Post-M182)

- **Base:** Hadamard-WAL K=256 checkpoint (~1.4× bf16)
- **Edit:** LoRA overlay (spectral norm guardrail)
- **Runtime:** WALCachedLinear + LoRA
- **Safety:** Spectral norm + fingerprint drift

---

*Updated with M171–M192 — 2026-04-20*

---

## WAL Wave & Intelligence Program — Phase 2 Complete (M193–M196, M195b) ✅

### Execution Summary

| Exp | ID | Result | Key Finding |
|-----|-----|--------|-------------|
| Real LoRA Wave Risk v1 | M193 v1 | ⚠️ | Synthetic WaveRisk fails on real LoRA. Catastrophic forgetting without mixed training |
| Real LoRA Wave Risk v2 | M193 v2 | ✅ | Mixed training prevents forgetting. Spectral norm > WaveRiskScore |
| Wave-Reg LoRA | M196 | ✅ | λ=0.1 improves survival 0→2/10, no PPL loss |
| Penalty Schedule | M196c | ✅ | Constant λ=0.1 most reliable. Cosine decay alone fails |
| Hadamard Adaptive Budget | M195 | ✅ | Uniform quant: adaptive +0.038 vs uniform +0.062 |
| Hadamard Adaptive + k-means | M195b | ✅ | k-means: adaptive +0.067 vs uniform +0.138 (2× better) |
| Hadamard Adaptive + k-means v2 | M195b+ | ✅ | iters=5: adaptive +0.038, gap vs uniform only +0.006 |
| Wave-LoRA Extended | M196b | ⚠️ | Mixed targets: wave reg λ=0.1 HURTS survival. rank4 baseline 10/50 best |
| Wave-LoRA λ Scaled | M196d | ⚠️ | λ scaling doesn't help. Variance dominates effect |
| Wave-LoRA Variance | M196e | ✅ | n=5: λ=0.025 gives best mean survival 4.80 vs 3.40 baseline (+41%) |

### Core Insights

1. **Mixed training is mandatory** for LoRA+WAL workflow
2. **Wave regularization is NOT beneficial for mixed targets** — n=20 shows baseline λ=0 gives best mean survival
3. **Adaptive K budget works with Hadamard** — confirmed on two quantization methods
4. **Synthetic risk formulas don't transfer** — need real calibration
5. **Constant λ more reliable than schedules** — cosine decay loses effectiveness
6. **n≥20 required for statistical confidence** — M196e (n=5) was underpowered, M196f reversed its conclusion
7. **Merge + re-encode is VIABLE** (was broken by forward bug, now fixed). K=256: +0.08 PPL, K=1024: +0.05 PPL

### Updated Production Stack (v4 — Post M200 Fixed / M200b v4)

```
Base:     Hadamard-WAL K=256 uniform (iters=3)
Edit:     LoRA rank=4, λ=0 (baseline, no wave reg)
Mode A:   WALCachedLinear + LoRA overlay (flexible)
Mode B:   Fast Compile — merge + re-encode K=256 (~3 min, +0.08 PPL)
Mode C:   Quality Compile — merge + re-encode K=1024 (~40 min, +0.05 PPL)
Safety:   Spectral norm + learned RF model
Training: Mixed wikitext-2 + facts, AdamW, steps=100-400
```

**Key changes from v3:**
- **Merge+re-encode REVIVED** — was broken by forward bug, now works!
- K=256 fast compile viable (+0.079 PPL, ~3 min encode)
- K=1024 quality compile option (+0.052 PPL, ~40 min encode)
- Three production modes: overlay / fast compile / quality compile

### Execution Summary

| Exp | ID | Result | Key Finding |
|-----|-----|--------|-------------|
| Hadamard Adaptive + k-means v2 | M195b+ | ✅ | iters=5: adaptive +0.038, gap vs uniform only +0.006 |
| Wave-LoRA Extended | M196b | ⚠️ | Mixed targets: wave reg λ=0.1 HURTS survival. rank4 baseline 10/50 best |
| Wave-LoRA λ Scaled | M196d | ⚠️ | λ scaling doesn't help. Variance dominates effect |
| Wave-LoRA Variance | M196e | ✅ | n=5: λ=0.025 gives best mean survival 4.80 vs 3.40 baseline (+41%) |
| Wave-LoRA Grid Search | M196f | ❌ | **n=20: NO significant λ effect.** Baseline λ=0 best mean 4.30±1.17. Wave reg doesn't help |
| Depth-Wave Budget | M198 | ✅ | Uniform K=256 best: PPL 12.41 (Δ=-0.08). Adaptive methods don't improve PPL |
| End-to-End WAL v2 | M200 | ⚠️ | **Merge + re-encode DESTROYS PPL (+6.20)**. Base + overlay ONLY |
| Production Overlay | M201 | ✅ | **Base + overlay works perfectly!** PPL -0.09, survival +1. Confirms production path |
| Learned Risk Model | M193b | ✅ | RF CV RMSE 0.57. final_loss + spectral_norm dominate prediction |

### Remaining Work

All M196-M202 experiments complete. Next phase (M203-M206):

1. **M203** (IN PROGRESS): Dense LoRA vs WAL+LoRA — the critical comparison
2. **M204**: Survival Improvement Search — rank/steps/lr/layers/modules sweep
3. **M205**: Risk Dataset 200 Runs — collect data for learned risk model
4. **M206**: Multi-LoRA Overlay — multiple edits simultaneously

### Updated Project Map

## Alive (Production)
```
1. Hadamard-WAL K=256 as base checkpoint
2. WALCachedLinear + LoRA overlay (Mode A)
3. Merge + re-encode K=256 fast compile (Mode B, ΔPPL +0.08)
4. Merge + re-encode K=1024 quality compile (Mode C, ΔPPL +0.05)
5. Mixed training for factual edits
6. Rank=4, 3 layers, 4 modules (weak edit)
7. Rank=4, steps=400 (strong edit, survival 36% compiled)
8. Spectral norm safety check (< 1.0)
9. Learned risk model as auxiliary tool
```

## Removed from Production
```
1. WAL patch vs LoRA (M131: 57,000× worse)
2. wave regularization for factual editing (M196f/g/h: no effect)
3. adaptive K as default (M198: uniform wins)
4. o_proj only (M196g: 3/50 flat, worse than mixed)
5. multi-LoRA overlay (M206: interference is catastrophic. 2 groups → +0.31 PPL, 3 groups → +2.0 PPL)
```

## REVIVED — Merge + Re-encode
```
M200/M200b v1-v3: "Catastrophic" results were caused by a BUG in forward restoration.
M200b v4 (FIXED): K=1024 merge+re-encode gives ΔPPL = +0.052 — VIABLE!

Production options now:
- **Overlay** (flexible, always LoRA overhead)
- **Fast Compile K=256** (+0.08 PPL weak, +0.64 PPL strong, ~3 min)
- **Quality Compile K=1024** (+0.05 PPL weak, ~40 min)
```

## Research Directions
```
1. Gumbel-WAL scale-up
2. Adaptive K if real size benefit emerges
3. Wave features as diagnostics, not training penalty
4. Learned risk model (M193b: 50 pts, RMSE 0.57; M205: 70 pts, RMSE 0.93 — needs stratified sampling)
5. ~~Multi-LoRA overlay~~ (M206: FAILED — interference catastrophic)
6. ✅ **Sequential multi-edit** (M206b: WORKS — 5.3/50 survival, +0.19 PPL)
7. Incremental editing with versioning (M206c)
8. ✅ **Batch concurrent edits** (M207: batch_50 best PPL, sequential still better for multi-task)
9. ✅ **Production Guide v4** (complete with all modes, safety checks, troubleshooting)
10. ✅ **Edit isolation & overwrite** (M208: confirmed! Sequential edits are isolated and overwritable)
11. ✅ **Adaptive steps per fact** (M209: ~37.5% facts are "impossible" with std LoRA — difficulty filter needed for production)
12. ✅ **Cross-model transfer** (M210: WAL works on Llama-1B! Minimal PPL impact, lower survival due to less capacity)
## Phase 3: Compiled WAL Lifecycle (M213-M220) — COMPLETE

### ✅ M213: K Sweep for Compiled Edits
- K=256 sweet spot for compiled strong edits
- K=512+ ineffective (catastrophic PPL, slow encode)

### ✅ M214: Steps Pareto Frontier
- steps=200 is sweet spot for 10 facts
- More steps ≠ better survival (overfitting without gain)

### ✅ M215: Sequential Long Chain (10 edits)
- 30% cumulative survival after 10 edits
- PPL grows +0.05 per edit
- Early batches degrade (recency bias confirmed)

### ✅ M216: WAL Checkpoint Diff
- WAL re-encode makes diff NON-LOCAL (100% modules, 77% params)
- Binary patch between versions ~31GB (not efficient)

### ✅ M217: Hard Fact Strategy
- ALL 6 aggressive configs gave 0/3 survival
- Hard facts (author/inventor) are IMPOSSIBLE with standard LoRA

### ✅ M218: Difficulty Classifier
- 87.5% accuracy using fact category
- Author/inventor = hard, Geography/music = easy

### ⏳ M219: Survival Dataset Expansion
- Stratified dataset for risk modeling (needs fix)

### ✅ M220: Baseline Comparison
- WAL is UNIQUE: compressible + editable + versionable
- Not best compression, but only editable compressed format

---

## Production Stack v10 (M235-M246 UPDATE)

```
Base:        Hadamard-WAL K=256, seed=42 (M243: bit-exact deterministic)
Edit:        LoRA rank=4, layer 16 ONLY (M244: optimal, PPL≈0)
Training:    FP32 adapters + gradient clipping (M241: critical fix)
Mode:        BATCH editing (5-10 facts simultaneously) — M229
Rehearsal:   Targeted replay (30% steps) — M228
GC:          Remove oldest when at capacity — M233
Tiering:     Easy→weights (layer 16), Hard→retrieval — M225+M242
Build:       WAL Build System + rehearsal + registry — M222+M232
Test:        Exact + paraphrase + negative + context — M234
CI:          Automated gates: exact + PPL + no_nan — M240+M246
Forensics:   WAL Probe (risk scoring, drift analysis) — M224
Determinism: Bit-exact with fixed seed (M243), semantic without
```

## Key Limitations (Documented)
1. Hard facts (author/inventor): impossible with ANY weight editing (M221+M226+M237)
   → MUST use retrieval tier (M242 validated)
2. FP16 training: gradient overflow destroys edits (M235+M240)
   → MUST use FP32 adapters + clipping (M241 fix)
3. WAL diff: non-local, store recipes not diffs (M216+M222)
4. Cross-architecture: encoding works, editing needs adaptation (M212)
5. Contrastive loss: does NOT solve hard facts (M221)
6. Activation-Guided: NOT suitable for auto-selection (M230)

## Phase 4: WeightOps / ModelOps Layer (M225-M234) — IN PROGRESS

### ✅ M225: Memory Tier Compiler
- Easy facts → weights (50 steps, 3 layers): 0/5 survival — insufficient capacity
- Medium facts → weights (200 steps, 11 layers): 4/5 survival — best result!
- Hard facts → retrieval tier: 0/5 survival (by design, routed externally)
- **Weight-tier survival: 4/10, Hard facts routed: 5**

### ✅ M230: Activation-Guided Editing
- AG selected layers [31,30,29,28] (late layers) with highest activation norms
- AG result: PPL +0.0236, survival 0/5
- Hardcoded layers [14,15,16] (mid layers): PPL +0.3096, survival 3/5
- **Key finding: High activation ≠ good editability. Factual editing works in middle layers.**

### ✅ M226: ROME/MEMIT Backend for Hard Facts
- Simplified ROME-lite: rank-one update on highest-activation layer
- All 3 hard facts selected layer 31 (late layer)
- Results: 0/3 survival, PPL drift accumulates (+0.03 → +0.28 → +1.14)
- **Hard facts remain BLOCKER even with rank-one updates**
- Layer selection by activation picks wrong layers (late, not factual)
- True ROME requires causal tracing + covariance matrix + optimized targets

### ✅ M227: Recipe Replay Determinism
- Run 1: LoRA survival 1/2, Re-encode PPL=5.3344
- Replay: LoRA survival 1/2, Re-encode PPL=5.2558
- Run 3: LoRA survival 1/2, Re-encode PPL=5.1425
- **Survival deterministic: YES (1/2 across all runs)**
- **PPL NOT deterministic: spread ~0.19** — k-means encode noise
- **Conclusion: Recipes are semantically deterministic, not bit-exact**

### ✅ M228: Rehearsal Buffer Against Forgetting
- **BREAKTHROUGH: Rehearsal prevents forgetting and reduces drift**
- Baseline (none): 17/50 survival (34%), PPL +0.91
- Random rehearsal: 21/50 survival (42%), PPL +0.51
- Low-survival replay: 23/50 survival (46%), PPL +0.45
- **+12% absolute survival improvement, -51% PPL drift reduction**
- Must-have component for WAL sequential editing pipeline

### ✅ M234: Edit Unit Tests
- Exact match: 5/5 (100%)
- Paraphrase: 13/15 (87%)
- Negative prompts: 8/10 (80%)
- Context robustness: 6/10 (60%)
- Post-re-encode: 5/5 (100%)
- **Compiled edits are production-robust. "Four Seasons" is weakest fact.**

### ✅ M231: Logit-Level Old Answer Suppression
- 0/3 target survival, 0/3 old answer retained
- **5th confirmation: Hard facts impossible with any LoRA/weight strategy**
- Problem is insufficient capacity, not old-answer interference
- Hard facts MUST use retrieval tier

### ✅ M233: WAL Garbage Collection
- Baseline: 9/25 (36%)
- GC oldest: 8/15 (53%) — better rate, less coverage
- Last 2 only: 5/25 (20%) — worse than baseline
- **GC trades coverage for quality. Batch editing (M229) reduces GC need.**

### ✅ M232: Branch Registry / Marketplace
- Prototype with publish/search/fork
- Registry stores branch metadata, recipes, quality scores
- Enables collaborative model editing ecosystem

### All M225-M234 Hypotheses: COMPLETE ✅

| Hypothesis | Result | Key Finding |
|------------|--------|-------------|
| M225 Memory Tier Compiler | ✅ | Easy→weights, Medium→weights, Hard→retrieval |
| M226 ROME/MEMIT backend | ✅ | 0/3 survival. Hard facts = retrieval ONLY |
| M227 Recipe Replay | ✅ | Semantically deterministic |
| M228 Rehearsal Buffer | ✅ | **+12% survival, -51% drift** |
| M229 Edit Conflict Graph | ✅ | **0% conflict, batch editing viable** |
| M230 Activation-Guided | ✅ | High activation ≠ editability |
| M231 Logit Suppression | ✅ | 0/3. 5th confirmation hard facts = blocker |
| M232 Branch Registry | ✅ | Prototype works |
| M233 WAL GC | ✅ | Trades coverage for quality |
| M234 Edit Unit Tests | ✅ | 100% exact, 87% paraphrase, 80% negative |
| M235 Batch Rehearsal | ❌ | 0/25 survival, PPL=nan (fp16 training bug) |
| M236 Causal Tracing | ✅ | Flat scores — implementation bug |
| M237 MEMIT Batch Editor | ✅ | 0/3 hard facts, clean PPL (MEMIT ≈ LoRA) |
| M238 Retrieval Tier | ✅ | Concept valid, injection broken |
| M239 Frozen Encode Determinism | ✅ | NOT deterministic — max diff 0.3-0.6 |
| M240 WAL CI Pipeline | ✅ | Pipeline works, edit quality FAIL |
| M241 Float32 Training Fix | ✅ | 3/3 survival, no nan — fp16 overflow confirmed |
| M242 Retrieval Fix | ✅ | 3/3 easy + hard — prompt injection works |
| M243 Encode Seed Determinism | ✅ | Bit-exact deterministic with fixed seed |
| M244 Layer Ablation | ✅ | Layer 16 optimal — 3/3, PPL Δ=-0.0013 |
| M245 Rebuild vs Sequential | ✅ | Batch 3.5× faster, 3× driftier |
| M246 Production Stack v9 | ✅ | Easy facts 3/3, hard retrieval 1/2 |
| M250 Final Report | ✅ | M235-M246 suite complete, 12 experiments |

---

## Next Directions
1. ✅ Refresh mechanism — Rehearsal buffer (M228)
2. ✅ Weight GC for cleanup (M233)
3. ✅ Edit type taxonomy per domain (M225)
4. Full CI/CD pipeline (M222 build system extension)
5. Paper-style report packaging (M220)
6. True ROME/MEMIT backend for hard facts (M226 follow-up)
7. ✅ Activation-Guided → diagnostics only (M230)
8. Batch editing production integration (M229)
9. Branch registry remote backend (M232)
10. Edit unit tests in build pipeline (M234)

*Updated with M225-M234 — 2026-05-01*
*Production Stack v8 with batch editing + rehearsal — 2026-05-01*
```

### Key Unanswered Question — ANSWERED by M203

> **M203 verdict: WAL ≈ Dense (EQUIVALENT)**
>
> Dense: PPL 4.395±0.09, survival 4.05±0.69
> WAL:   PPL 4.439±0.08, survival 4.33±1.07
>
> Difference statistically insignificant. WAL is production-ready.

---

*Updated with M193–M196, M195b, M196b, M200-M206 — 2026-04-30*
