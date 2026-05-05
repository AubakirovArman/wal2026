# WAL Grammar Draft v0.1
## Weight-Aligned Language: A Programming Language for Neural Network Weights

**Status:** Draft  
**Based on:** M45 prototype + M46 full-model validation  
**Date:** 2026-04-20

---

## 1. Motivation

Neural network weights are treated as opaque arrays of floats. Quantization methods replace floats with integers or codes, but these are still static tables. WAL (Weight-Aligned Language) treats each weight as a **program** that executes at runtime to reconstruct its value.

Why a language?
- **Compositionality**: A weight can be a sum of multiple atoms, not just a lookup.
- **Interpretability**: Programs can be analyzed (e.g., "which atoms dominate?").
- **Precision control**: `lmax` (max instructions) is a tunable quality knob.
- **Hardware efficiency**: Small instruction sets map well to SIMD/GPU execution.
- **Meta-learning**: Programs can be edited, merged, or evolved.

---

## 2. Core Abstraction: The Atom

An **atom** is a learned basis function. In WAL-0 (Scalar), an atom is simply a float value. In WAL-1 (Vector), an atom is a vector. In WAL-2 (Matrix), an atom is a small matrix.

Atoms are **shared** across all weights in a parameter. They are learned via k-means (or other dictionary learning) on the weight distribution.

```
AtomTable[K] for parameter P
  atom[k] : float32  (WAL-0)
```

---

## 3. WAL-0: Scalar Grammar

The simplest WAL dialect. Each weight executes a program of atom lookups and additions.

### 3.1 Instruction Set (ISA-0)

| Opcode | Name | Operands | Stack Effect | Description |
|--------|------|----------|--------------|-------------|
| `0x00` | `PUSH_ATOM` | `idx: u8` | `→ atom[idx]` | Push atom[k] onto stack |
| `0x01` | `MUL` | `coeff: i8` | `a → a*coeff` | Multiply top by coefficient |
| `0x02` | `ADD` | — | `a b → a+b` | Pop two, push sum |
| `0x03` | `STOP` | — | `a → (return a)` | Halt, top of stack is result |
| `0x04` | `NOP` | — | `→` | No-op (for padding/alignment) |

**Notes:**
- Coefficients in ISA-0 are typically ternary: `{-1, 0, +1}`.
- `MUL 0` is equivalent to skipping the atom (NOP).
- `STOP` is implicit at program end if not present.

### 3.2 Program Format

A program is a sequence of instructions. For compact storage:

```
program := {instruction}*
instruction := opcode [operand]
```

**Packed encoding (binary):**
```
Bits [1:0]   = opcode (0-3)
Bits [7:2]   = operand (6 bits for idx or coeff)
```

For K=128, idx fits in 7 bits. With 2-bit opcode, each instruction is 9 bits — pack 7 instructions into 8 bytes (with 1-bit overhead) or use 2 bytes per instruction (simple).

**Simpler format (M46 implementation):**
```python
program = [(idx_1, coeff_1), (idx_2, coeff_2), ...]
# Stored as two parallel arrays:
indices:  uint8[K]   # atom indices
signs:    int8[K]    # {-1, 0, +1} coefficients
```

### 3.3 Example Programs

**DRL v2 (static lookup):**
```
PUSH_ATOM 42
STOP
```
Equivalent to: `weight = atom[42]`

**WAL Scalar lmax=2 (M46):**
```
PUSH_ATOM 7
MUL +1
PUSH_ATOM 23
MUL -1
ADD
STOP
```
Equivalent to: `weight = atom[7] * (+1) + atom[23] * (-1)`

**WAL Scalar with zero coefficient (sparse):**
```
PUSH_ATOM 5
MUL +1
PUSH_ATOM 12
MUL 0
ADD
STOP
```
Equivalent to: `weight = atom[5]` (second term skipped)

### 3.4 Execution Model

**Interpreter (CPU/GPU):**
```python
def execute(program, atom_table):
    stack = []
    for op, operand in program:
        if op == PUSH_ATOM:
            stack.append(atom_table[operand])
        elif op == MUL:
            stack[-1] *= operand
        elif op == ADD:
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif op == STOP:
            break
    return stack[-1]
```

**SIMD/GPU Kernel:**
For large tensors, the interpreter loop is unrolled into a CUDA kernel:
```cuda
// Each thread executes one weight's program
__global__ void wal_scalar_kernel(
    const uint8_t* indices,    // [N, lmax]
    const int8_t*  signs,      // [N, lmax]
    const float*   atoms,      // [K]
    float*         output,     // [N]
    int N, int lmax
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = 0.0f;
    for (int s = 0; s < lmax; s++) {
        int k = indices[i * lmax + s];
        int c = signs[i * lmax + s];
        acc += atoms[k] * c;
    }
    output[i] = acc;
}
```

**Complexity:** O(lmax) per weight. For lmax=2, this is 2 loads + 2 FMAs — negligible overhead vs static lookup.

---

## 4. WAL-1: Vector Grammar (Future)

Instead of scalar atoms, use vector atoms. Each row of a weight matrix is reconstructed as a weighted sum of vector atoms.

```
atom[k] : float32[D]  # D-dimensional vector
weight[row] = Σ_s atom[k_s] * c_s
```

Benefits:
- Fewer atoms needed (vector atoms capture correlations across columns)
- Better compression for tall matrices
- Natural extension of scalar WAL

---

## 5. WAL-2: Tensor Block Grammar (Future)

Atoms are small tensors (e.g., 4×4 blocks). Each block of the weight matrix is reconstructed from tensor atoms.

```
atom[k] : float32[B, B]  # B×B block
weight[block_i, block_j] = Σ_s atom[k_s] * c_s
```

This bridges to VQ-VAE and neural compression techniques.

---

## 6. Stop Bit & Variable Length

Not all weights need lmax instructions. Some converge earlier.

**Variable-length encoding:**
```
program := instruction* STOP
instruction := (idx: u7, coeff: i2)  # packed into 9 bits
```

Or use **stop depth** per weight (as in M43-M44 route encoding):
```python
stop_depth[i] = number of active instructions for weight i
# 0 = weight is zero (all coeffs = 0)
# 1 = weight = atom[k1] * c1
# 2 = weight = atom[k1] * c1 + atom[k2] * c2
```

This is how M46 internally works — `lmax=2` is max, but many weights use fewer.

---

## 7. Grammar BNF

```bnf
<program>       ::= <instruction>* <stop>
<instruction>   ::= <atom_call> | <arith>
<atom_call>     ::= "ATOM" <idx> <coeff>
<arith>         ::= "ADD"
<stop>          ::= "STOP" | ε
<idx>           ::= 0 .. K-1
<coeff>         ::= -1 | 0 | +1
```

**Sugar:**
```
<weight_expr>   ::= <term> ("+" <term>)*
<term>          ::= <coeff> "*" "atom[" <idx> "]"
<coeff>         ::= "+" | "-" | "0"
```

Example: `+atom[7] -atom[23]` → `weight = atom[7] - atom[23]`

---

## 8. Runtime: WAL VM

The WAL Virtual Machine is a minimal execution environment:

```
Registers:
  ACC : float32      # accumulator
  PC  : uint32       # program counter
  SP  : uint32       # stack pointer (optional)

Memory:
  ATOM_TABLE[K]     # shared read-only atoms
  PROGRAM[N, lmax]  # per-weight programs
  OUTPUT[N]         # reconstructed weights

Execution loop (per thread/weight):
  ACC = 0
  for step in 0..lmax-1:
    k = PROGRAM[idx, step].idx
    c = PROGRAM[idx, step].coeff
    if c != 0:
      ACC += ATOM_TABLE[k] * c
    if PROGRAM[idx, step].stop:
      break
  OUTPUT[idx] = ACC
```

**No stack needed for WAL-0** — a single accumulator suffices.

---

## 9. From DRL to WAL: Paradigm Shift

| Aspect | DRL v2 | WAL Scalar |
|--------|--------|-----------|
| Model | Static lookup table | Dynamic program |
| Per-weight | 1 center ID | lmax atom calls |
| Storage | Codebook + IDs | Atom table + programs |
| Expressiveness | O(K) | O(K^lmax * 3^lmax) |
| Runtime | `table[id]` | `Σ atom[k_s]*c_s` |
| Quality (K=128) | PPL 4.29 | PPL 2.78 |
| Composability | None | Yes (atoms sum) |
| Interpretability | Opaque | Programs analyzable |

---

## 10. Open Questions

1. **Atom sharing across layers?** Currently atoms are per-parameter. Can we share atoms across similar layers?
2. **Learned coefficients?** Ternary {-1,0,+1} is simple. Would learned floats (e.g., 3-bit) help?
3. **lmax adaptation per layer?** Early layers might need lmax=1, late layers lmax=3.
4. **JIT compilation?** Can we compile WAL programs to custom CUDA kernels?
5. **Higher-order WAL?** Can programs call other programs or use control flow?

---

## 11. Next Steps

1. Implement WAL-0 GPU kernel in Triton
2. Measure encode/decode throughput vs static lookup
3. Explore WAL-1 (vector atoms) for row-wise compression
4. Design compact binary format for model shipping
5. Experiment with cross-layer atom sharing
