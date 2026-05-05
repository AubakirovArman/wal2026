# M62: WAL v2 Grammar & Assembler Round-Trip Validation

## Date
2026-04-20

## Goal
Build and validate the text syntax, parser, assembler, and disassembler for WAL v2. Prove exact round-trip: binary → text → binary.

## What Was Built

### 1. BNF Grammar (`src/wal/v2/grammar.py`)
```bnf
<program_stream>   ::= <header> <program>*
<header>           ::= "K" <uint> "C" <uint> "SHAPE" <uint> <uint>
<program>          ::= <atom_call> [<residual>]
<atom_call>        ::= "ATOM" <atom_id> "COEF" <float>
<residual>         ::= "RESIDUAL" <float>
```

### 2. Text Format v0.1
```wal
; WAL v2 v0.1 — 67,108,864 programs
K 256
C 16
SHAPE 8192 8192

ATOM 120 COEF 0.771360
ATOM 70 COEF -0.716460
ATOM 90 COEF -0.716460
```

### 3. Assembler (`src/wal/v2/asm.py`)
- **Input**: WAL text + AtomTable + CoeffTable
- **Output**: ProgramBufferV2
- **Quantization**: coeff_value → nearest coeff_id via `np.abs(coeff_values - value).argmin()`
- **Validation**: K/C match, shape match, program count match

### 4. Disassembler (`src/wal/v2/asm.py`)
- **Input**: ProgramBufferV2 + AtomTable + CoeffTable
- **Output**: WAL text
- **Modes**: `full` (all programs) or `unique` (summary with counts)

## Validation Results

### Test 1: Text Preview (first 20 programs)
- Disassemble time: <1ms
- Human-readable output confirmed

### Test 2: Unique Program Summary
- **Disassemble time**: 272.8s (for 67M weights)
- **Unique programs**: 1,299 / 67,108,864 (0.0019%)
- Top program: `ATOM 120 COEF 0.771360` — 297,809 occurrences

### Test 3: Round-Trip (10K sample)
- Disassemble: 0.035s
- Assemble: 0.062s
- **atom_ids match**: True
- **coeff_ids match**: True
- **residuals match**: True
- **has_residual match**: True
- **→ ROUND-TRIP PASS**

### Test 4: Reconstruction Match
- **Exact match**: True
- **Max diff**: 0.0000000000
- Assembled buffer produces bit-identical reconstruction

## What This Achieves

1. **WAL is no longer a black-box codec** — programs are inspectable and editable in any text editor.
2. **Lossless text representation** — exact round-trip proves the grammar is complete and unambiguous.
3. **Vocabulary visibility** — 1,299 unique programs for 67M weights reveals the "language" of this layer.
4. **Foundation for ecosystem** — parser enables: VM, debugger, linter, optimizer, version control.

## Artifacts
- `src/wal/v2/grammar.py`
- `src/wal/v2/asm.py`
- `experiments/m62_wal_v2_grammar_asm.py`
- `experiments/m62_wal_v2_grammar_asm.log`

## Next Steps
- Phase 3: WAL VM + Runtime — formal execution model, reference interpreter, Triton kernel
- Phase 4: Compression Format v2 — binary serialization for model shipping


## Extracted Metrics (from source)

- Time: .3
- Time: .3
