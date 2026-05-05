# 13 — Errors and Lessons: The Complete Taxonomy

> *"We didn't fail 59 times. We discovered 59 ways that don't work."*

This chapter collects every major error, false assumption, and wrong turn from the entire project (M1-M83) into a single reference. Read this before starting any WAL-related work.

---

## Error Category 1: Metric Delusion

### E1.1 — relMSE is Meaningless

**When**: M43 (VRE), M71 (single-layer PPL)

**What happened**: VRE showed relMSE 0.001 and correlation 0.9992 — seemingly perfect. Full-model PPL: >7000.

**Why**: Per-layer metrics measure local reconstruction error. They do not measure how errors propagate through 80 layers of attention, normalization, and residual connections. A small structured error can be amplified exponentially.

**Rule**: *relMSE is permanently discarded. Only full-model PPL is valid.*

---

### E1.2 — Single-Layer PPL is Not Predictive

**When**: M71

**What happened**: Single-layer PPL delta +0.0002. Full-model PPL delta +4.90. Difference: **24,500×**.

**Why**: Error accumulation is nonlinear. A layer that looks harmless in isolation can be the straw that breaks the camel's back when combined with 79 other slightly-perturbed layers.

**Rule**: *Always test full-model PPL. Single-layer metrics are at best weakly correlated.*

---

### E1.3 — Output Correlation is Misleading

**When**: M43

**What happened**: Output correlation 0.9992 — "the outputs are almost identical."

**Reality**: Correlation measures linear relationship, not absolute error. Two tensors can have correlation 1.0 while differing by a constant offset that breaks softmax normalization.

**Rule**: *Correlation without absolute error bounds is meaningless.*

---

## Error Category 2: The 12-Bit Hard Floor

### E2.1 — Anything Less Than 12 Bits/Weight Destroys Quality

**When**: M69-M73 systematic sweep

**Data**:
| K (atoms) | Bits/weight | PPL |
|-----------|-------------|-----|
| 16 | 4 | 111,000 |
| 32 | 5 | 1,500 |
| 64 | 6 | 46.6 |
| 128 | 7 | 7.68 |
| 256 | 8 | 3.02 (degraded) |

**Baseline**: 2.7805

**Why**: The degradation is caused by structured error accumulation across 80 layers. Each layer adds a small perturbation. After 80 layers, the perturbations compound.

**Rule**: *12 bits/weight is a physical constraint, not a temporary limitation. Treat it as such.*

---

### E2.2 — Two-Tier PQ at 8 Bits Still Fails

**When**: M73

**What happened**: Two-tier product quantization at 8 bits achieved PPL 3.1137 — still degraded.

**Lesson**: Even sophisticated quantization schemes cannot beat the 12-bit floor. The problem is not the quantization method — it's the bit budget.

---

## Error Category 3: Spatial Correlation Kills

### E3.1 — Block-Quantization is Architecturally Toxic

**When**: M43

**What happened**: VRE (block-quantization) with relMSE 0.001 produced PPL >7000.

**Why**: Block-quantization introduces spatially-correlated errors. Attention softmax is highly sensitive to spatially-correlated perturbations because it operates on rows. A consistent error across a block shifts the entire softmax distribution.

**Contrast**: Scalar per-weight encoding introduces independent noise. The law of large numbers smooths this out across a row. Block errors do not smooth out.

**Rule**: *Never introduce spatially-correlated errors into weights. Per-weight independence is a hard requirement.*

---

## Error Category 4: Structure Cannot Be Mined Post-Hoc

### E4.1 — Grammar Mining on Flat Encodings Fails

**When**: M56, M27 WAL-SS through WAL-E2E

**What happened**: Every attempt to mine grammar, templates, macros, or structure from flat greedy encodings produced negligible coverage (<0.05%) and degraded quality.

**Why**: Greedy residual encoding produces an i.i.d. stream. There is no structure to mine because the encoding process does not create structure.

**Rule**: *Structure must be enforced at ISA level, not mined post-hoc.*

---

### E4.2 — Explicit Grammar Without Semantic Constraints is Useless

**When**: M27 WAL-FG

**What happened**: A formal grammar with 15 rules over 4 phrase slots produced parse trees, but rule coverage was 0.000272 and quality degraded from PPL 2.4081 to 2.6866.

**Why**: The grammar was syntactically valid but semantically empty. It constrained form, not meaning.

**Rule**: *Grammar without semantic constraints is decoration, not structure.*

---

## Error Category 5: Vector Atoms are Not Ready

### E5.1 — Ternary Coefficients Cannot Represent Vectors

**When**: M49-M50

**What happened**: WAL-1 vector atoms with ternary {-1,0,+1} and lmax=2 produced relMSE 0.08-0.99 — catastrophic.

**Why**: A vector of dimension D needs D coefficients. Ternary with 2 steps provides at most 2 effective coefficients. For D>2, this is hopelessly underparameterized.

**Rule**: *Do not use vector atoms until the coefficient representation problem is solved.*

---

## Error Category 6: Implementation Bugs That Look Like Algorithm Failures

### E6.1 — The Compare Harness Bug

**When**: Step 16

**What happened**: `eager-bf16` accidentally received `shape_policy_json`, corrupting all operational comparisons for weeks.

**Impact**: Every strategy comparison was wrong. Decisions were made on bad data.

**Lesson**: *Audit harness integrity before drawing conclusions. A buggy benchmark is worse than no benchmark.*

---

### E6.2 — The Codebook Indexing Bug

**When**: M57

**What happened**: `codebook_recon` indexed by `unique_prog` order but `program_ids` by `sort_idx`. Result: PPL **299,002**.

**Fix**: One line. But finding it took hours.

**Lesson**: *Always verify index consistency when sorting/reordering. Write assertions.*

---

### E6.3 — uint8 as Boolean Mask

**When**: Multiple occasions

**What happened**: PyTorch treats `uint8` scalar as boolean mask, returning shape `[8]` instead of single scalar.

**Fix**: Always convert uint8 to `int()` before indexing.

**Lesson**: *PyTorch's uint8 behavior is a footgun. Always be explicit.*

---

### E6.4 — GPU Context Switch Bug

**When**: M53c

**What happened**: `invalid resource handle` when Triton kernel launched on different GPU than tensor.

**Fix**: Explicit `torch.cuda.device(device)` wrapper.

**Lesson**: *Multi-GPU code must be device-explicit. Never assume default device.*

---

## Error Category 7: Runtime Illusions

### E7.1 — Fused Kernels Are Not Automatically Faster

**When**: Step 9, M53b

**What happened**: Fused Triton kernel was correct but slower than reference packed path.

**Why**: Memory bandwidth is the bottleneck, not compute. Fusing decode into matmul doesn't help if you're still memory-bound.

**Lesson**: *Profile before optimizing. The bottleneck is rarely where you think it is.*

---

### E7.2 — Hot-Route Caching Fails on Flat Distributions

**When**: Step 10

**What happened**: Top-128 routes cover only 17.61% of weights. Not enough to matter.

**Why**: Neural network weight distributions are surprisingly uniform. No natural "hot set."

**Lesson**: *Caching requires skew. If the distribution is flat, caching is waste.*

---

### E7.3 — Persistent Palettes Are Unstable

**When**: Step 12 (Variants A, B0-B3)

**What happened**: Every variant of persistent palette either produced NaN, CUDA asserts, or was 2.5-10× slower than baseline.

**Why**: Per-row dynamic LUT gather inside inner loop is inherently unstable. Branch-heavy kernels on GPUs are slow.

**Lesson**: *GPU kernels hate branches. Prefer uniform access patterns.*

---

## Error Category 8: Over-Engineering

### E8.1 — Dynamic Depth Added Complexity Without Quality Gain

**When**: Step 1

**What happened**: Dynamic depth (L_max=12) seemed elegant but added massive complexity. The actual win came from better calibration, not dynamic depth.

**Lesson**: *Start simple. Add complexity only when simple fails.*

---

### E8.2 — The Ladder Architecture Was Over-Engineered

**When**: Entire DRL v2 era

**What happened**: Ladders, routes, palettes, stages, tiles, hot prefixes, register files — a whole vocabulary of concepts. Almost none survived.

**Lesson**: *Every concept you add is a concept you must debug, test, and maintain. Prefer deletion.*

---

## The Meta-Lessons

### ML1 — Early Bad Results Don't Falsify the Idea

The first real-tensor failure (Step 2) could have killed the project. Instead, the team recognized the calibration was wrong, not the ternary-route math. This established a critical methodology: **implementation bugs look like algorithm failures.**

### ML2 — Define Gates Early

The WikiText-2 gate (Step 6) established "good enough" as PPL gap <0.01. After that, effort pivoted to runtime. Without this gate, the project would have chased quality forever.

### ML3 — Operational vs Research Branches

The compare harness bug (Step 16) forced a clean separation. Operational code must be conservative. Research code can be experimental. Never let them contaminate each other.

### ML4 — The Simplest Program That Works

After 59 experiments, the only program that consistently worked was `atom × coeff + residual`. Everything more complex failed. This became WAL's design philosophy: **simplicity is not a compromise — it's a requirement.**

### ML5 — Document Failures More Than Successes

Success teaches you what works on one path. Failure teaches you what doesn't work on any path. The prehistory chapter (M1-M59) is the most valuable documentation in this project because it prevents repetition.
