# Phase 2: Grammar & Assembler (M62)

## The Problem

WAL programs were binary blobs. You couldn't read them, edit them, or diff them. If WAL is a language, it needs a human-readable form.

## What Was Built

- **BNF grammar**: Formal grammar for WAL program syntax
- **Parser**: Text → ProgramBufferV2
- **Pretty-printer**: ProgramBufferV2 → text
- **Assembler/Disassembler**: Full bidirectional conversion

## Text Format Example

```wal
K 256
C 16
SHAPE 8192 8192

ATOM 0 = 0.123456
ATOM 1 = -0.789012

ATOM 120 COEF 0.771360
ATOM 45 COEF 1.234567 RESIDUAL 0.001234
```

## Key Results

| Metric | Value |
|--------|-------|
| Round-trip | **Exact** (max error 0.00) |
| Unique programs / layer | ~1,299 per 67M weights |
| Vocabulary ratio | 0.0016% |

## Why This Matters

The text format makes WAL programs inspectable. You can grep for specific atoms. You can diff two encodings. You can hand-edit a program. This is the difference between a opaque binary format and a true language.

## Files
- `src/wal/v2/grammar.py`
- `src/wal/v2/asm.py`
- `experiments/m62_wal_v2_grammar_asm.py`
