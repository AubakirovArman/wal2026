# 01 — Prehistory: M1–M59, Every Failed Path

> *"The first 59 experiments were a graveyard of beautiful ideas. This chapter is the headstones."*

## Overview

Before WAL existed, there was DRL v2 — Dynamic Route Language version 2. The goal was the same (compress neural network weights) but the approach was completely different: ternary routes, ladders, palettes, and stages. Fifty-nine experiments explored every corner of this design space. Almost everything failed.

This chapter documents every major failed approach, why it failed, and what we learned. The purpose is not to celebrate failure but to **prevent repetition**. Every idea here seemed reasonable at the time. Many produced excellent single-layer metrics. All failed at full-model scale.

---

## The DRL v2 Architecture

DRL v2 encoded each weight as a **ternary route** — a sequence of `{-1, 0, +1}` decisions through a decision tree ("ladder"). The ladder was calibrated per row. Each decision multiplied the current value by a learned scale. After `L_max` steps (initially 5, later 12), the route stopped.

The encoding had several moving parts:
- **Ladders**: per-row calibrated decision trees
- **Routes**: per-weight ternary sequences
- **Stages**: groups of 256 routes sharing a codebook
- **Tiles**: 16×16 blocks of weights processed together
- **Palettes**: learned codebooks of route segments

This architecture was elegant. It was also catastrophically complex.

---

## Step 0: Restart (Archive Route B)

The project began by archiving all previous work (Route B) into `archiv.org/` and `bk/`. The key decision: **quality, not compression, is the priority**. This seems obvious now but was a hard-won insight — the previous project had been chasing bits-per-weight at the expense of model quality.

**Lesson**: Before building, decide what you're optimizing for. WAL optimizes for PPL gap vs dense, not bpw.

---

## Step 1: Dynamic-Depth Ternary Routes (L_max = 12)

**What we tried**: Variable-length ternary routes up to depth 12, with per-weight `stop_depth`.

**Why it seemed good**: Fixed-depth routes (L_max=5) hit a ceiling on long context because outlier rows couldn't extend locally. Dynamic depth addresses both: short routes for normal weights, long routes for outliers.

**What happened**: On synthetic data, it looked great. On real tensors, it produced catastrophic errors.

**Root cause**: The ladder calibration was wrong. Not the math — the calibration recipe. The ladder fit drifted away from row-normalized top outliers.

**Fix**: Pin the top step to preserve exact coverage of the largest weight in every row. Collapse real-tensor relMSE from percent-level to `1e-5` scale.

**Lesson**: *Don't blame the math — check the calibration recipe first.* This became the project's most important methodological principle.

---

## Step 2: First Real-Tensor Probes — The Calibration Crisis

**What we tried**: Apply the ternary route encoder to real Llama 70B weights.

**What happened**: Catastrophic errors. relMSE at percent level. The project nearly died here.

**Root cause**: The ladder calibration assumed normalized inputs but real weight rows have extreme outliers. The geometric seed and 20 CD iterations worked for synthetic data but not for real distributions.

**Fix**: Row norm + geometric seed + pinned top step + 20 CD iterations. This calibration recipe collapsed relMSE to `1e-5`.

**Lesson**: *Synthetic success does not predict real-tensor success. Always validate on real model weights immediately.*

---

## Step 3: Working Calibration

**What we tried**: The fixed calibration recipe on real tensors.

**What happened**: relMSE `1e-5`. Quality confirmed.

**Lesson**: Sometimes the fix is simple — once you know what to fix.

---

## Step 4: Two-Pass Encoding — Routes → Codebook → IDs

**What we tried**: Encode every weight to a route, then build a codebook of unique routes and store an ID per weight.

**Why it seemed good**: The set of routes that actually occurs is tiny compared to `3^12`. An ID tensor + shared codebook decouples "language design" from "storage".

**What happened**: ~1500 unique routes per tensor. ~11 bits/weight. This was the first compression result that actually worked.

**Lesson**: Codebook entropy is a diagnostic. If your codebook has 1500 entries from 67M weights, your "language" has a tiny vocabulary.

---

## Step 5: Full-Model Encode Sweep (560 Tensors)

**What we tried**: Encode all 560 weight tensors in Llama 70B.

**What happened**: Mean relMSE `3.91e-6`. Quality confirmed across the full model.

**Lesson**: Single-layer success can generalize — but only if you test it.

---

## Step 6: WikiText-2 Gate — The Quality Pivot

**What we tried**: Full WikiText-2 perplexity evaluation.

**Result**: Dense PPL 3.4304 vs routed PPL 3.4350 — gap only 0.0046.

**Why this was a turning point**: After this result, the project's open front definitively shifted from quality to runtime. Quality was "good enough." Speed was the new problem.

**Lesson**: Define your gates early. If you don't know what "good enough" means, you'll chase perfection forever.

---

## Step 7: HumanEval Smoke + Full

**What we tried**: HumanEval benchmark (164 tasks) on encoded model.

**Result**: Dense 0.7317 vs routed 0.7195 — gap -0.0122.

**Lesson**: Coding benchmarks are more sensitive to weight perturbation than language modeling. But the gap is still small.

---

## Step 8: Runtime Becomes the Bottleneck

**What we tried**: Measure inference speed of encoded model.

**What happened**: Small-batch latency was poor due to weight materialization. The encoded model had to decode weights on every forward pass.

**Lesson**: Compression without fast decode is useless for inference. The decoder matters as much as the encoder.

---

## Step 9: Fused Triton Kernel — Negative Result

**What we tried**: A fused Triton kernel that decodes routes directly during matrix multiplication.

**What happened**: Correct but **slower** than the reference packed path. Removing materialization ≠ speed win.

**Lesson**: Fused kernels are not automatically faster. Memory bandwidth is the bottleneck, not compute.

---

## Step 10: Hot-Route Shortcut — Negative Result

**What we tried**: Cache the most common routes (hot-route shortcut) to avoid full decode.

**What happened**: Route distribution too flat. Top-128 routes cover only **17.61%** of weights. Not enough to matter.

**Lesson**: Caching only works when the distribution is skewed. Neural network weights are surprisingly uniform.

---

## Step 11–12m: Runtime-Aware Distillation

**What we tried**: Tile-local palette, grouped execution, shape policies, bounded cached-packed.

**What happened**: A long sequence of experiments:
- Tile-local palette works for quality
- Grouped execution needed for speed
- Local path finally beats fused global
- Runtime pivot to cached-packed

**Lesson**: The right runtime strategy is context-dependent. There's no universal "fastest" decode — it depends on batch size, sequence length, and layer shape.

---

## Step 12: RouteDistill Prototype — Partial Success

**What we tried**: Distill the runtime behavior of routes into a faster representation.

**What happened**: Tile-local palette works for quality. But grouped multi-tile, 2D grouping, and shape policies were all needed to get speed.

**Lesson**: Quality and speed are separate problems. Solve quality first, then optimize speed.

---

## Step 16: The Compare Harness Bug

**What we tried**: Compare different encoding strategies using an automated harness.

**What happened**: `eager-bf16` accidentally received `shape_policy_json`, corrupting all operational comparisons. Every comparison for weeks was wrong.

**Root cause**: A configuration leak between test branches.

**Fix**: Clean separation between operational (`eager-bf16`/`eager-hybrid`) and research branches.

**Lesson**: *Audit harness integrity before drawing operational conclusions.* A buggy benchmark is worse than no benchmark.

---

## M20: Fast Reconstruct — New Baseline

**What we tried**: A new fast reconstruction baseline.

**What happened**: New packed baseline established. This became the reference for all subsequent runtime comparisons.

---

## M21–M24: Stage-Drop Experiments — Dead End

**What we tried**: Drop less important stages to save compute.

**What happened**: PPL 3.4761 — **dead**. Too crude a global lever.

**Lesson**: Global heuristics fail. Per-layer decisions are needed.

---

## M23–M25: Grammar + Same-Encoding Runtime

**What we tried**: Mine grammar from route streams and build same-encoding runtime.

**What happened**: Stage-local hot/cold confirmed as exact win. But grammar mining on flat greedy encodings does not work.

**Lesson**: Post-hoc structure mining fails. Structure must be enforced at ISA level.

---

## M43: The VRE Catastrophe — The Most Important Failure

**What we tried**: Block-quantization (VRE — Vector Route Encoder) on Llama 70B.

**Single-layer metrics**: relMSE 0.001, output correlation 0.9992 — perfect.

**Full-model result**: PPL > **7000**.

**Why**: Block-correlated errors act as adversarial perturbation to attention softmax. Scalar's independent per-weight noise is tolerated. VRE's spatially-correlated errors are catastrophic.

**Consequence**: This failure killed vector-route encoding permanently. It proved that **single-layer metrics are completely unreliable predictors of full-model behavior**. The difference can be up to **24,500×**.

**Lesson**: *relMSE is meaningless. Only full-model PPL matters.* This lesson was so important that it became a project commandment.

---

## M44: WAL Language Inception

**What we tried**: A complete reframing.

**Old framing**: "Discrete weight quantization" — minimize bits, preserve quality.
**New framing**: "Weight Assembly Language" — weights are programs, encoding is disassembly, runtime is execution.

**Why this mattered**: It opened the grammar/induction research line (M27) and established WAL as a language-design project, not a compression project.

---

## M45: WAL Scalar Prototype

**What we tried**: The simplest possible WAL program: `weight = atom × coeff`.

**Result**: **200× better relMSE than DRL v2** at same K=128.

**Lesson**: Simplicity wins. The simplest program that works is better than a complex program that almost works.

---

## M46: Full 70B WAL Scalar PPL

**What we tried**: Encode full Llama 70B with WAL scalar.

**Result**: PPL **2.7821** vs dense 2.7805 — gap +0.06%.

**Lesson**: WAL scalar works at scale. The simple program preserves quality.

---

## M49–M50: WAL-1 Vector Atoms — Catastrophic Failure

**What we tried**: Use vector atoms instead of scalar atoms. Each atom is a vector, not a scalar.

**Result**: relMSE **0.08–0.99**. Catastrophic failure.

**Why**: Ternary `{-1,0,+1}` with `lmax=2` fundamentally insufficient for high-dimensional vectors.

**Consequence**: WAL-1 vector atoms were permanently abandoned. WAL-0 scalar became the only proven foundation.

**Lesson**: The coefficient representation problem must be solved before vector atoms can work.

---

## M51: Compile-Time Specialization — No Win

**What we tried**: Specialize the kernel at compile time for K=128.

**Result**: No win. Generic kernel already near memory bandwidth.

**Lesson**: Don't optimize what's not the bottleneck.

---

## M52: Cross-Layer Atom Sharing

**What we tried**: Share atoms across layers instead of per-layer atoms.

**Result**: Shared atoms beat per-layer up to **7.7×**.

**Lesson**: Distribution-level atoms (one table per model family) are promising. This became a future research direction.

---

## M53: Fused Triton Encode — 309× Speedup

**What we tried**: Fused Triton kernel for encoding.

**Result**: **309× kernel speedup**. Full 70B encode in 2225s.

**Lesson**: GPU-native encoding is critical. CPU loops are the enemy.

---

## M54: Codebook Mining + Decode — 1.1 TW/s

**What we tried**: Mine unique programs from the encoded weights and build a codebook.

**Result**: 1,079 unique programs from 67M weights. Decode at **1.1 TW/s** (teraweights/second) via single `index_select`.

**Lesson**: The vocabulary of weight programs is tiny. This is the compression secret: not fewer bits per weight, but shared structure across weights.

---

## M55: Variable Length / Early Stopping

**What we tried**: Allow programs to stop early if residual is small enough.

**Result**: Works but threshold must be <0.002 for quality.

**Lesson**: Variable length is possible but the threshold is sensitive.

---

## M56: Grammar Analysis — The Brutal Truth

**What we tried**: Analyze WAL-0 programs for grammatical structure.

**Result**: WAL-0 programs form an **i.i.d. stream** with:
- No spatial autocorrelation
- No n-gram patterns
- No grammar

**Lesson**: Scalar greedy residual encoding does not create linguistic structure. Hierarchy and constraints must be enforced at ISA level.

---

## M57: Full 70B Codebook Encode — 6× Faster

**What we tried**: Full 70B encode using codebook approach.

**Result**: PPL 2.7828 in **437s** — 6× faster than M46.

**Bug found**: `codebook_recon` indexed by `unique_prog` order but `program_ids` by `sort_idx` → PPL **299,002** before fix.

**Lesson**: Always verify index consistency when sorting/reordering. A simple indexing bug can produce absurd results.

---

## M58–M59: Global Codebook — Fast Encode

**What we tried**: Global codebook across all layers.

**Result**: Fast encode confirmed.

---

## Summary of Prehistory

### What Worked
- Scalar ISA (`atom × coeff + residual`)
- Two-pass encoding (k-means + Lloyd-Max)
- GPU-native encoding (Triton)
- Codebook mining (tiny vocabulary)

### What Failed
- Vector atoms (catastrophic relMSE)
- Block-quantization (VRE: PPL >7000)
- Grammar mining on flat encodings (no structure)
- Runtime shortcuts (hot-route, fused kernels)
- Global heuristics (stage-drop)
- Compile-time specialization (no win)

### The Key Insight
After 59 experiments, the project had converged on a single truth:

> **The simplest program that works is `atom × coeff + residual`.**

Everything more complex failed. This became the foundation for all 11 phases.
