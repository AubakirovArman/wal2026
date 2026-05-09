# WAL v2 / Global Research Program

**Date:** 2026-04-20
**Status:** Planning document
**Premise:** WAL v1 proved weights can be tokenized. WAL v2 must verify: can we choose a weight representation space where tokens become more stable, compact, meaningful, and useful for editing?

---

## Core Thesis

> WAL v1 доказал: веса можно токенизировать.
> WAL v2 должен проверить: можно ли выбрать такое пространство представления весов, где токены становятся стабильнее, компактнее, осмысленнее и полезнее для редактирования.

---

## Phase 1 — Frozen Vocabulary Core

**Goal:** Make frozen atom/coeff table the official WAL mode.

M133 showed the diffuse diff problem was not WAL itself, but atom table rebuild. With frozen table, global diff dropped from **25% to 0.168%**, target layers stayed modified, and non-target layers became **0% diff**.

```
WAL-Frozen:
  build atom/coeff table once on base model
  freeze vocabulary
  encode base
  edit
  re-encode edited with same vocabulary
  compare/apply/diff programs
```

**To verify:**
1. frozen table PPL penalty
2. frozen table across many LoRA edits
3. frozen table across style/domain/unlearning edits
4. frozen table across rank/steps sweep
5. frozen table on 8B / 14B / 70B

**Main hypothesis:** Atom/coeff table is a tokenizer of weights. If tokenizer is frozen, WAL becomes comparable and diffable.

---

## Phase 2 — WAL Patch v2

M131 showed naive WAL patch was huge — **10.7 GB vs 0.19 MB LoRA**. But M133 after frozen table reduced estimate to ~**75 MB**, because diff localized to target layers only.

The goal is not to "beat LoRA in size", but to understand:

> Can WAL patch become a standalone structural patch format?

**To verify:**
1. patch apply: WAL_base + WAL_patch → WAL_edited
2. patch correctness: PPL, target accuracy, generation behavior
3. patch compression: RLE positions, bitmask per tensor, transition table, coeff-only patch, atom-only patch, block patch, entropy coding
4. patch stacking: patch A + patch B, conflict resolution, rollback

**Target:** 75 MB → 10–30 MB

LoRA will still be smaller. But WAL patch can be useful as:
- immutable compiled edit
- runtime-free edit
- program-space audit trail
- deployment artifact without adapter logic

---

## Phase 3 — WAL+LoRA Overlay as Product Path

M135 showed LoRA can train directly on top of `WALCachedLinear`, without full decode→dense cycle. WAL+LoRA trained gave **10/10**, merge back to WAL also preserved **10/10**.

This is currently the most practical path.

**Goal:** Make official architecture:

```
Base:     model.wal12
Overlays: edit_001.lora
          edit_002.lora
          edit_003.lora

Runtime:
  load WAL
  unpack 12-bit
  decode/cache dense weights
  apply selected LoRA overlays
  optional merge → new WAL
```

**To verify:**
1. multiple LoRA overlays simultaneously
2. enable/disable edits on the fly
3. merge multiple LoRAs into WAL
4. latency overhead
5. memory overhead
6. conflicting LoRAs
7. LoRA router per task

**Main hypothesis:** WAL should not replace LoRA. WAL should become a structural base checkpoint, and LoRA — a small editable overlay.

---

## Phase 4 — Re-Encode Geometry

M138 showed: rank=4 is sweet spot, rank=1 is dangerous at 200 steps, and survival without PPL can be misleading.

Now we need to go deeper: not just rank/steps, but **update geometry**.

**Goal:** Understand which LoRA updates are compatible with WAL re-encode.

**Metrics to compute:**
1. ||ΔW||_F
2. spectral norm ΔW
3. max(abs(ΔW))
4. mean(abs(ΔW))
5. kurtosis / heavy-tailness
6. boundary crossing rate
7. atom-change %
8. coeff-change %
9. both-change %
10. target/non-target diff
11. KL to base model
12. preserve loss

**Goal:** Create WAL Edit Safety Score

```
if ΔW too sharp → high risk
if boundary crossing > threshold → stop
if KL grows faster than target accuracy → stop
```

**Main hypothesis:** WAL-edit stability depends not only on rank and steps, but on the shape of update relative to quantization cells.

---

## Phase 5 — Spectral / Transform-WAL

New major direction: **weights as a frozen wave**.

Take a layer as a frozen signal:
```
W[layer] = final form after training
```

Decompose it:
```
Ŵ(u,v) = Σ Σ W(x,y) e^{-2πi(ux/M + vy/N)}
```

**Goal:** Check if LLM weight matrices have useful frequency representation.

### 5.1 Spectral Probe
```
FFT2 / DCT2 / Hadamard per layer
energy spectrum
top-k frequency energy
low/mid/high frequency ratio
```
Compare: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, early/mid/late layers, base vs edited.

### 5.2 Spectral Truncation PPL
```
keep top 1%, 5%, 10%, 25% frequencies
inverse transform
replace layer
full-model PPL
```
Only PPL as gate. Not MSE.

### 5.3 Transform-WAL
```
W → Transform(W) → WAL encode → WAL decode → inverse Transform → W_recon
```
Check:
- DCT-WAL
- FFT-WAL
- Hadamard-WAL
- Random Orthogonal-WAL

### 5.4 Spectral-WAL Diff
```
raw WAL diff = ? %
spectral WAL diff = ? %
```

**Main hypothesis:** Raw weights may be a bad space for WAL. Frequency/transform space may make programs more stable and diff more local.

---

## Phase 6 — Wave-Atom ISA

Not just transform before WAL, but a new language version.

Current WAL:
```
weight = atom[id] × coeff[id] + residual
```

Wave-WAL:
```
weight[i,j] = amplitude[id] × wave(freq_id, phase_id, i, j) + residual
```

Or sum of waves:
```
W[i,j] = Σ A_k cos(ωx_k i + ωy_k j + φ_k) + residual[i,j]
```

**Goal:** Build new ISA where atom is not a scalar value, but a wave generator.

**To verify (small scale):**
1. 1 layer
2. 1 module
3. small model
4. top-K wave atoms
5. PPL, runtime, size

Risky, but these are the things that can give a new WAL v2/v3.

---

## Phase 7 — Graph-WAL

Important angle: ordinary Fourier assumes row/column order has meaning. But in LLM, hidden channels may be permutable. Therefore ordinary FFT may not be the right space.

Idea:
> Weights live not on a regular grid, but on a graph of neuron/channel similarity.

**Goal:**
```
build channel graph
graph Fourier transform
encode weights in graph spectral space
```

**Graph construction:**
1. cosine similarity rows
2. cosine similarity columns
3. activation correlation
4. attention head grouping
5. neuron co-activation

Then:
```
graph Laplacian
eigenvectors
graph Fourier coefficients
WAL encode coefficients
```

**Main hypothesis:** If hidden dimensions have no natural order, graph Fourier may be more correct than ordinary FFT.

---

## Phase 8 — Semantic Fingerprints v2

M137 gave early positive result: WAL fingerprints sense model state — small delta for seed, moderate for noise, large for K change.

Now we need a real benchmark.

**Goal:** WAL fingerprints as model forensics.

**Models to collect:**
- base
- instruct
- code-tuned
- math-tuned
- medical-tuned
- safety-tuned
- overfit-LoRA
- merged-LoRA
- noisy
- quantized

**Features:**
- atom entropy
- coeff entropy
- top-k dominance
- residual density
- program transition matrix
- layer-wise KL
- module-wise fingerprint
- spectral fingerprint
- patch fingerprint

**Target:** classifier fingerprint → model type, accuracy >80%

This can become a separate direction: **WAL forensic diagnostics**.

---

## Phase 9 — Cross-Model Frozen Vocabulary

Global atoms work inside one model. Next question:

> Can we make a frozen vocabulary for a model family?

**Goal:**
```
Llama-3.x tokenizer of weights
Qwen tokenizer of weights
Gemma tokenizer of weights
```

**To verify:**
1. build atom/coeff table on Llama 8B
2. encode Llama 70B using same table
3. build on 70B, encode 8B
4. build on base, encode instruct
5. build on instruct, encode base

**Metrics:** PPL, program entropy, atoms used, patch locality, fingerprint stability.

**Main hypothesis:** A model family may have its own weight-tokenizer.

If this works, WAL becomes much more interesting.

---

## Phase 10 — WAL Training / WAL-Friendly Models

So far WAL is mostly post-hoc: take a trained dense model and encode.

But if we want real semantics/structure, perhaps the model should be trained with WAL constraints.

**Goal:** Train a small model that is WAL-friendly from the start.

**Variants:**
1. Soft-WALLinear
2. Gumbel-softmax atom/coeff ids
3. STE hard program ids
4. regularizer on stable atom usage
5. regularizer on sparse program transitions
6. frozen vocabulary during training

**Target size (start small):**
```
10M → 100M → 300M → 1B
```

**Main hypothesis:** Post-hoc WAL gives basis atoms. WAL-aware training may produce a more organized program grammar.

---

## 10 Global Tracks (Summary)

```
Track 1:  Frozen Vocabulary Core
Track 2:  WAL Patch v2
Track 3:  WAL+LoRA Overlay Product Path
Track 4:  Re-Encode Geometry
Track 5:  Spectral / Transform-WAL
Track 6:  Wave-Atom ISA
Track 7:  Graph-WAL
Track 8:  Semantic Fingerprints
Track 9:  Cross-Model Vocabulary
Track 10: WAL-Friendly Training
```

---

## Priorities

### Immediate (practical foundation)
```
1. Frozen table PPL penalty
2. WAL patch apply + patch compression
3. WAL+LoRA overlay multi-edit
4. Re-Encode Geometry / Safety Score
```
This strengthens current WAL.

### Medium-term (new scientific branch)
```
5. Spectral Probe
6. Transform-WAL
7. Spectral-WAL diff
8. Wave-Atom ISA prototype
```
This checks the "weights as frozen wave" idea.

### Long-term (deep theory)
```
9. Graph-WAL
10. WAL-friendly training
```
This can become a new major research area.

---

## Recommended Next Experiment

If choosing one global experiment:

### Transform-WAL Benchmark

Because it connects everything:
```
WAL × Fourier/Hadamard/DCT × diff stability × compression × PPL × patch locality
```

**Matrix:**
```
Raw-WAL
DCT-WAL
FFT-WAL
Hadamard-WAL
RandomOrthogonal-WAL
```

**Metrics:**
```
PPL
re-encode ΔPPL
program diff %
target/non-target locality
patch size
fingerprint separability
encode/decode speed
```

**Success criterion:** If Transform-WAL improves at least 2 of 5:
```
PPL
diff locality
patch size
fingerprint clarity
re-encode stability
```
then this is a new major path.

---

## Strategic Conclusion

WAL already has a working core:
```
canonicalization
frozen vocabulary
12-bit packing
LoRA overlay
localized diff
fingerprints
```

The next leap should not be "another small patch test", but a search for **a better weight representation space**.

The wave idea is very timely:

> Perhaps current WAL tokenizes weights in the wrong space.
> Spectral / transform / graph space may be the space where weight-programs become more stable, compact, and meaningful.

This is the global task of WAL v2.
