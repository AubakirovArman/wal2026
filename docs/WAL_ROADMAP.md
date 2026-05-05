# WAL Roadmap: From Runtime to Language
## What We Have vs What We Need

**Last updated:** 2026-04-20  
**Status after M47-M52**

---

## ✅ FOUNDATION — Already Built

### 1. WAL-0 Scalar ISA
- **Instructions:** `PUSH_ATOM`, `MUL`, `ADD`, `STOP`
- **Storage:** `ProgramBuffer` (parallel arrays: indices + signs)
- **Packing:** `pack_programs` / `unpack_programs` (compact int16 encoding)
- **Location:** `src/wal/isa.py`

### 2. Encoder
- **Algorithm:** Greedy residual encoding with k-means++ atoms
- **Quality:** relMSE ~1e-5 (lossless for 70B PPL)
- **Speed:** ~0.1-0.3s per parameter (GPU k-means, 1M samples)
- **Location:** `src/wal/encoder.py`

### 3. Decoder (Runtime)
- **PyTorch:** CPU/GPU compatible
- **Triton:** 406 Mweights/sec on H200
- **Correctness:** Verified vs PyTorch (max error < 1e-5)
- **Location:** `src/wal/decoder.py`, `src/wal/triton_kernels.py`

### 4. Binary Format
- **Magic:** `WAL0`
- **Header:** 32 bytes (version, K, lmax, N, dtype)
- **Payload:** Atom table + programs + metadata JSON
- **Round-trip:** Verified on real 70B layer (relMSE 4.5e-6)
- **Location:** `src/wal/format.py`

### 5. End-to-End Validation
- **M46:** Full 70B model → PPL 2.7821 vs baseline 2.7805 (+0.06%)
- **M48:** Real layer round-trip → output correlation 1.000000
- **M52:** Cross-layer shared atoms → 2-7× better than per-layer

---

## 🔧 NEAR-TERM — Needed for Production WAL

### 1. Compression (M53)
**Problem:** WAL-0 currently uses 2 bytes/weight (0.50× compression vs bf16)

**Solutions:**
- [ ] **Uint8 packing** for K ≤ 85: 1 byte/weight → 1.00×
- [ ] **Codebook deduplication:** Top-256 programs cover 99.9% weights → store unique programs in codebook, per-weight stores 1-byte ID
- [ ] **Variable-length encoding:** Store `stop_depth` per weight (many weights use lmax=1 or 0)
- [ ] **Row-scale quantization:** Store row scales in fp16 or 8-bit log scale

**Expected result:** 1.5-2.0× compression with zero PPL loss

### 2. Cross-Layer Shared Atoms (M55)
**Problem:** Currently atoms are per-parameter (540 tables for 70B model)

**Solutions:**
- [ ] Implement global atom table builder (pool samples from all layers)
- [ ] Group layers by similarity (early vs late, attention vs MLP)
- [ ] Measure full-model PPL with shared atoms

**Expected result:** Fewer atom tables, slightly better quality (M52 showed shared > per-layer)

### 3. Human-Readable Syntax (WAL Text Format)
**Problem:** WAL is binary-only. No way to inspect or edit programs.

**Solutions:**
- [ ] Design text syntax: `weight = +atom[7] -atom[23];`
- [ ] Parser: text → ProgramBuffer
- [ ] Pretty-printer: ProgramBuffer → text
- [ ] Assembler/Disassembler in `src/wal/asm.py`

**Example:**
```wal
; WAL-0 program for weight i
ATOM 7, +1
ATOM 23, -1
STOP
```

### 4. WAL VM (Formal Execution Model)
**Problem:** No formal VM spec. Kernel is ad-hoc.

**Solutions:**
- [ ] Define VM state: PC, ACC, atom table, program memory
- [ ] Instruction cycle: FETCH → DECODE → EXECUTE
- [ ] Reference interpreter in Python
- [ ] Triton kernel as "VM accelerated mode"

**Location:** New file `src/wal/vm.py`

### 5. Integration with PyTorch
**Problem:** WAL-encoded weights are not usable as `nn.Parameter`

**Solutions:**
- [ ] `WALParameter` class (subclass of `nn.Parameter`)
- [ ] Custom autograd function for forward/backward through WAL decode
- [ ] Hook into `transformers` model loading (`from_pretrained` → WAL decode)

---

## 🔮 LONG-TERM — Research Directions

### 6. WAL-1 Vector / Tensor Block (M54 v2)
**Problem:** M50 showed ternary lmax=2 fails for vector atoms in high-D

**Possible approaches:**
- [ ] **Continuous coefficients:** 3-bit or 4-bit learned coefficients instead of {-1,0,+1}
- [ ] **Autoencoder atoms:** Train atoms via neural autoencoder, not k-means
- [ ] **Tensor block atoms:** 4×4 or 8×8 blocks (bridge to neural compression)
- [ ] **Higher lmax:** lmax=8-16 for vector atoms (quality vs speed tradeoff)

### 7. Control Flow (WAL-2)
**Problem:** WAL-0 is purely linear. No conditionals, loops, or functions.

**Ideas:**
- [ ] `IF weight > threshold: use_atom_set_A ELSE use_atom_set_B`
- [ ] `REPEAT lmax_times: find_best_atom`
- [ ] `CALL subroutine` for shared sub-programs

**Use case:** Adaptive precision — early layers get more atoms, late layers fewer.

### 8. Meta-Learning / Program Evolution
**Problem:** Programs are fixed after encoding.

**Ideas:**
- [ ] Fine-tune programs (not weights) via gradient descent
- [ ] Evolve atoms for downstream tasks
- [ ] Merge programs from multiple models (model soup at program level)

### 9. Hardware-Specific Backends
**Problem:** Only CUDA/Triton supported.

**Backends needed:**
- [ ] Metal (Apple Silicon)
- [ ] ROCm (AMD)
- [ ] CPU SIMD (AVX-512, NEON)
- [ ] WebGPU (browser inference)

### 10. WAL Standard Library
**Problem:** Every model trains its own atoms from scratch.

**Ideas:**
- [ ] Pre-trained atom libraries for common architectures (Llama, GPT, Mistral)
- [ ] Atom zoo: curated sets optimized for different domains (code, math, multimodal)
- [ ] Atom transfer: use atoms from Llama-70B for Llama-8B

---

## Priority Matrix

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P0 | Compression (uint8 + codebook) | Medium | **Critical** — without compression WAL is not deployable |
| P0 | Cross-layer shared atoms + full PPL test | Medium | **Critical** — proves scalability |
| P1 | Text syntax + assembler | Low | High — developer experience |
| P1 | WAL VM spec + reference interpreter | Medium | High — foundation for ecosystem |
| P1 | PyTorch integration (`WALParameter`) | Medium | High — usability |
| P2 | WAL-1 vector v2 (continuous coeffs) | High | High — unlocks massive compression |
| P2 | Control flow (WAL-2) | High | Medium — research value |
| P3 | Hardware backends | High | Medium — portability |
| P3 | Meta-learning / fine-tune programs | High | High — but unproven |
| P3 | Standard library / atom zoo | Low | Medium — ecosystem |

---

## Definition of Done: "WAL is a Language"

WAL becomes a true language when:

1. ✅ **Syntax exists** — human can write `weight = +atom[7] -atom[23]`
2. ✅ **Runtime exists** — VM executes programs at 100+ Mw/s
3. ✅ **Compiler exists** — encoder transforms weights → programs
4. 🔲 **Serializer exists** — binary format for shipping (M47 done, but needs compression)
5. 🔲 **Debugger exists** — can inspect programs, step through execution
6. 🔲 **Standard library exists** — pre-trained atoms for common models
7. 🔲 **Ecosystem exists** — integrations with PyTorch, HF, ONNX

**Current score: 4/7** 🟡
