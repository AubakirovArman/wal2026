# WAL — The Book

> A complete history of the Weight Assembly Language project: what was tried, what failed, what worked, and why.

## What This Book Is

This book documents the full journey of WAL — from its origins as a failed dynamic-route compression system, through 13 phases of iterative refinement, to a production-ready framework for encoding neural network weights as programs.

It is written for:
- **Future maintainers** who need to understand why WAL works the way it does
- **Researchers** who want to build on WAL without repeating our mistakes
- **Engineers** who want to use WAL in production

## How to Read This Book

| Path | For |
|------|-----|
| **Quick start** → `00_PROLOGUE.md` + `15_FUTURE.md` | Executives, PMs |
| **Technical deep dive** → `02_PHASE_01.md` through `12_PHASE_11.md` | Engineers implementing WAL |
| **Lessons learned** → `01_PREHISTORY.md` + `13_ERRORS_AND_LESSONS.md` | Researchers building on WAL |
| **Full narrative** → Read in order 00 → 15 | Historians, future maintainers |

## Table of Contents

### Part I: Foundations
- [`00_PROLOGUE.md`](00_PROLOGUE.md) — What WAL is and why it exists
- [`01_PREHISTORY.md`](01_PREHISTORY.md) — M1-M59: every failed path (DRL v2, Route B, WAL-0 through WAL-LDI)

### Part II: The 13 Phases
- [`02_PHASE_01_ENCODER.md`](02_PHASE_01_ENCODER.md) — Phase 1: Encoder v2 (M60-M61)
- [`03_PHASE_02_GRAMMAR.md`](03_PHASE_02_GRAMMAR.md) — Phase 2: Grammar & Assembler (M62)
- [`04_PHASE_03_VM.md`](04_PHASE_03_VM.md) — Phase 3: VM + Runtime (M63)
- [`05_PHASE_04_COMPRESSION.md`](05_PHASE_04_COMPRESSION.md) — Phase 4: Compression (M64)
- [`06_PHASE_05_HIERARCHY.md`](06_PHASE_05_HIERARCHY.md) — Phase 5: Hierarchical Atoms (M65-M75)
- [`07_PHASE_06_PYTORCH.md`](07_PHASE_06_PYTORCH.md) — Phase 6: PyTorch Integration (M76-M77)
- [`08_PHASE_07_DEBUGGER.md`](08_PHASE_07_DEBUGGER.md) — Phase 7: Debugger (M78)
- [`09_PHASE_08_STDLIB.md`](09_PHASE_08_STDLIB.md) — Phase 8: Standard Library (M79)
- [`10_PHASE_09_BACKENDS.md`](10_PHASE_09_BACKENDS.md) — Phase 9: Hardware Backends (M80)
- [`11_PHASE_10_META.md`](11_PHASE_10_META.md) — Phase 10: Meta-Learning (M81-M82)
- [`12_PHASE_11_ECOSYSTEM.md`](12_PHASE_11_ECOSYSTEM.md) — Phase 11: Ecosystem (M83)
- [`16_PHASE_12_KV_CACHE.md`](16_PHASE_12_KV_CACHE.md) — Phase 12: KV-cache WAL (M84-M88)
- [`17_PHASE_13_STREAMING.md`](17_PHASE_13_STREAMING.md) — Phase 13: Streaming Encoder (M89-M90)
- [`18_PHASE_14_QAT.md`](18_PHASE_14_QAT.md) — Phase 14: QAT for WAL (M91-M95)
- [`20_PHASE_15_HYBRID_WORKFLOW.md`](20_PHASE_15_HYBRID_WORKFLOW.md) — Phase 15: Hybrid LoRA→WAL Workflow (M110)

### Part IV: Determinism & Stability (M126-M129)
- [`31_PHASE_27_REENCODE_STABILITY.md`](31_PHASE_27_REENCODE_STABILITY.md) — Phase 27: Re-Encode Stability Matrix (M128)
- [`32_PHASE_28_CANONICALIZATION.md`](32_PHASE_28_CANONICALIZATION.md) — Phase 28: Canonicalization Layer (M129)
- [`33_PHASE_16_v4_REPRODUCIBILITY_GATE.md`](33_PHASE_16_v4_REPRODUCIBILITY_GATE.md) — Phase 16 v4: Reproducibility Gate with Canonicalization (M126)
- [`34_PHASE_33_RUNTIME_BENCHMARK.md`](34_PHASE_33_RUNTIME_BENCHMARK.md) — Phase 33: Runtime Benchmark (M132)
- [`35_PHASE_19_v2_CAUSAL_WAL_PATCH.md`](35_PHASE_19_v2_CAUSAL_WAL_PATCH.md) — Phase 19 v2: Causal WAL Patch (M130)
- [`36_PHASE_29_EDIT_COMPILATION.md`](36_PHASE_29_EDIT_COMPILATION.md) — Phase 29: Edit Compilation (M131)
- [`37_PHASE_A_FIXED_ATOM_TABLE.md`](37_PHASE_A_FIXED_ATOM_TABLE.md) — Phase A: Fixed Atom Table (M133)
- [`38_PHASE_C_WAL_LORA_OVERLAY.md`](38_PHASE_C_WAL_LORA_OVERLAY.md) — Phase C: WAL+LoRA Overlay (M135)
- [`39_PHASE_D_12BIT_PACKING.md`](39_PHASE_D_12BIT_PACKING.md) — Phase D: 12-bit Packing (M136)
- [`40_PHASE_F_SEMANTIC_FINGERPRINTS.md`](40_PHASE_F_SEMANTIC_FINGERPRINTS.md) — Phase F: Semantic Fingerprints (M137)
- [`41_PHASE_G_REENCODE_LOSS.md`](41_PHASE_G_REENCODE_LOSS.md) — Phase G: Re-Encode Loss Characterization (M138)
- [`42_M139_WAL_PATCH_V2.md`](42_M139_WAL_PATCH_V2.md) — Track 2: WAL Patch v2 (M139)
- [`43_M140_WAL_LORA_MULTI.md`](43_M140_WAL_LORA_MULTI.md) — Track 3: WAL+LoRA Overlay Multi-Edit (M140)
- [`44_M141_REENCODE_GEOMETRY.md`](44_M141_REENCODE_GEOMETRY.md) — Track 4: Re-Encode Geometry / Safety Score (M141)
- [`45_M142_TRANSFORM_WAL_PROBE.md`](45_M142_TRANSFORM_WAL_PROBE.md) — Track 5: Transform-WAL Probe (M142)
- [`46_M143_WAVE_ATOM_ISA.md`](46_M143_WAVE_ATOM_ISA.md) — Track 6: Wave-Atom ISA Probe (M143)
- [`47_M144_GRAPH_WAL_PROBE.md`](47_M144_GRAPH_WAL_PROBE.md) — Track 7: Graph-WAL Probe (M144)
- [`48_M145_SEMANTIC_FINGERPRINTS_V2.md`](48_M145_SEMANTIC_FINGERPRINTS_V2.md) — Track 8: Semantic Fingerprints v2 (M145)
- [`49_M146_CROSS_MODEL_VOCAB.md`](49_M146_CROSS_MODEL_VOCAB.md) — Track 9: Cross-Model Frozen Vocabulary (M146)
- [`50_M147_WAL_FRIENDLY_TRAINING.md`](50_M147_WAL_FRIENDLY_TRAINING.md) — Track 10: WAL-Friendly Training (M147)
- [`51_WAL_v2_GLOBAL_SUMMARY.md`](51_WAL_v2_GLOBAL_SUMMARY.md) — WAL v2 Global Program: Complete 10-Track Summary
- [`51_M148_WAL_v1_SPEC_FREEZE.md`](51_M148_WAL_v1_SPEC_FREEZE.md) — WAL v1 Spec Freeze (M148)
- [`52_M154_FIX_HADAMARD.md`](52_M154_FIX_HADAMARD.md) — Fix Hadamard Properly (M154)
- [`53_M169_WAL_ABLATION_DASHBOARD.md`](53_M169_WAL_ABLATION_DASHBOARD.md) — WAL Ablation Dashboard (M169)
- [`54_M168_STANDARD_BENCHMARK.md`](54_M168_STANDARD_BENCHMARK.md) — Standard WAL Benchmark Suite (M168)
- [`55_M149_M153_PARTIAL_NOTES.md`](55_M149_M153_PARTIAL_NOTES.md) — M149/M153 Timeout Notes
- [`56_M149_FROZEN_VOCAB_PPL_MATRIX.md`](56_M149_FROZEN_VOCAB_PPL_MATRIX.md) — Frozen Vocab PPL Matrix (M149)
- [`57_M153_TRANSFORM_WAL_ENCODER.md`](57_M153_TRANSFORM_WAL_ENCODER.md) — Transform-WAL Encoder (M153)
- [`58_M156_TRANSFORM_WAL_DIFF_LOCALITY.md`](58_M156_TRANSFORM_WAL_DIFF_LOCALITY.md) — Transform-WAL Diff Locality (M156)
- [`59_M152_SAFETY_SCORE_REAL_LORA.md`](59_M152_SAFETY_SCORE_REAL_LORA.md) — Safety Score on Real LoRA (M152)
- [`60_M150_LORA_PATCH_COMPRESSION.md`](60_M150_LORA_PATCH_COMPRESSION.md) — LoRA Patch Compression (M150)
- [`61_M160_SPECTRAL_ENERGY_MAP.md`](61_M160_SPECTRAL_ENERGY_MAP.md) — Spectral Energy Map (M160)
- [`62_M151_MULTI_LORA_ROUTING.md`](62_M151_MULTI_LORA_ROUTING.md) — Multi-LoRA Routing (M151)
- [`63_M157_TRANSFORM_VOCAB_STUDY.md`](63_M157_TRANSFORM_VOCAB_STUDY.md) — Transform Vocabulary Study (M157)
- [`64_M161_SPECTRAL_DELTA_LORA.md`](64_M161_SPECTRAL_DELTA_LORA.md) — Spectral Delta of LoRA (M161)
- [`65_M159_TRANSFORM_METADATA_COST.md`](65_M159_TRANSFORM_METADATA_COST.md) — Transform Metadata Cost (M159)

### Part III: Reflections
- [`13_ERRORS_AND_LESSONS.md`](13_ERRORS_AND_LESSONS.md) — The complete error taxonomy
- [`14_GLOSSARY.md`](14_GLOSSARY.md) — WAL terminology
- [`15_FUTURE.md`](15_FUTURE.md) — What comes next

## Key Statistics

| Metric | Value |
|--------|-------|
| Total experiments | ~170+ (M1-M129) |
| Failed approaches documented | 20+ |
| Phases completed | 18/18 |
| Tests passing | 65/65 |
| Lines of source code | ~3,500 |
| Lines of diary entries | ~2,000+ |
| Model size validated | Llama 3.3 70B (70 billion parameters) |
| Best PPL achieved | 2.7781 (vs baseline 2.7805) |
| Compression ratio | 1.33× vs bf16 |

## WAL v2 Global Program

The next phase of WAL research is defined in [`WAL_v2_GLOBAL_PROGRAM.md`](../WAL_v2_GLOBAL_PROGRAM.md):

> WAL v1 proved weights can be tokenized. WAL v2 must verify: can we choose a weight representation space where tokens become more stable, compact, meaningful, and useful for editing?

**10 Global Tracks:** Frozen Vocabulary → WAL Patch v2 → LoRA Overlay → Re-Encode Geometry → Spectral-WAL → Wave-Atom ISA → Graph-WAL → Fingerprints → Cross-Model Vocab → WAL-Friendly Training.

### Track Summaries
- [`WAL_v2_TRACKS_1_3_SUMMARY.md`](../WAL_v2_TRACKS_1_3_SUMMARY.md) — Tracks 1–3: Frozen Vocabulary, WAL Patch v2, LoRA Overlay

## The Central Thesis

> **Weight = Program**
>
> Every weight in a neural network can be represented as a short program that multiplies an atom by a coefficient. The atoms are learned. The programs are discrete. The decode is fast. The quality is dense-level.

This sounds simple. It took 95 experiments to make it true.
