# Phase 4: Compression (M64)

## The Problem

Text format is human-readable but bloated. Binary format is needed for shipping.

## What Was Built

- **WAL2 binary format**: Magic header + uint4-packed coefficients + sparse residual bitmap
- **Streaming structure**: Can decode partial files
- **Bit-exact round-trip**: serialize → deserialize == original

## Format Structure

```
[Magic: "WAL2"]
[Version: 2]
[Header: K, C, N, M, flags]
[Atom table: K floats]
[Coeff table: C floats]
[Programs: N atom_ids (uint8) + N coeff_ids (uint4 packed)]
[Residual bitmap: N bits]
[Residual values: sparse float16]
```

## Key Results

| Metric | Value |
|--------|-------|
| Format size | ~12 bits/weight |
| Round-trip | **Bit-exact** |
| Streaming | Supported |

## Why This Matters

Without binary format, WAL is a research toy. With binary format, WAL is a shipping format.

## Files
- `src/wal/v2/format.py`
- `experiments/m64_compression.py`
