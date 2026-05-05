# Phase 23: WAL Size Benchmark

> *"WAL does not compress. WAL structures."*

## Status

**Phase:** 23  
**Goal:** Compute the exact size of WAL-encoded Llama 3.1 8B and compare to baselines.  
**Date:** 2026-04-20  
**Method:** M123 — theoretical calculation + single-layer validation  
**Result:** ✅ **COMPLETE** — WAL = 11.26 GB (12 bits/weight). Not compression — editability.

## The Numbers

### Model Stats
- **Total params:** 8.03B
- **Linear params:** 7.505B (weights that get encoded)
- **Other params:** 0.525B (embeddings, norms — stay in original format)

### Baseline Sizes

| Format | Size | Ratio vs bf16 |
|--------|------|---------------|
| fp32 | 32.12 GB | 2.0× |
| bf16 | 16.06 GB | 1.0× |
| int8 | 8.03 GB | 2.0× smaller |
| int4 | 4.02 GB | 4.0× smaller |

### WAL Size Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Atom table | 262 KB | 256 atoms × 256 floats (fp32) |
| Coeff table | 64 bytes | 16 coeffs × 4 bytes |
| Programs (byte-aligned) | 15.01 GB | 2 bytes/weight (uint8 atom_id + uint8 coeff_id) |
| Programs (packed 12-bit) | **11.26 GB** | 8b atom_id + 4b coeff_id |
| **WAL total (packed)** | **11.26 GB** | — |

### Compression Ratios

| vs Format | Ratio | Interpretation |
|-----------|-------|----------------|
| vs bf16 | **1.43×** | WAL is 43% larger |
| vs int8 | **1.87×** | WAL is 87% larger |
| vs int4 | **3.74×** | WAL is 274% larger |

## Key Insight

**WAL is NOT a compression method.**

It is a **structural representation** that trades size for:
1. **Editability** — programs can be inspected, diffed, and manipulated
2. **Determinism** — encode/decode is fully specified
3. **Mergeability** — models can be merged via weight-space soups then re-encoded
4. **Inspection** — program heatmaps show weight structure

If you need compression, use int8 or int4. If you need structural editability, use WAL.

### The Trade-Off

```
Compression ←————————————————→ Editability
   int4          WAL           dense
  4 GB        11.3 GB        16 GB
  smallest      structured     largest
  no edit       editable       fully editable
```

## Files

- `experiments/m123_wal_size_benchmark.py`

## Next Steps

Phase 24: Do layers share atoms?
