# M168 — Standard WAL Benchmark Suite

**Date:** 2026-04-20
**Status:** ✅ Complete
**Goal:** Define unified JSON output format for all WAL experiments.

## Schema

```json
{
  "experiment": "string (required)",
  "model": "string (required)",
  "mode": "string (required)",
  "ppl": "float | null",
  "mse": "float | null",
  "relmse": "float | null",
  "max_err": "float | null",
  "patch_size_bytes": "int | null",
  "patch_size_mb": "float | null",
  "lora_size_mb": "float | null",
  "diff_target": "float [0,1] | null",
  "diff_nontarget": "float [0,1] | null",
  "encode_time_sec": "float | null",
  "decode_time_sec": "float | null",
  "inference_speedup": "float | null",
  "atom_entropy": "float | null",
  "coeff_entropy": "float | null",
  "program_entropy": "float | null",
  "safety_score": "string | null",
  "spectral_norm": "float | null",
  "status": "string: complete | partial | negative | failed",
  "error": "string | null",
  "notes": "string | null",
  "timestamp": "string | null"
}
```

## Validation Rules

- `ppl`, `mse`, `relmse`, `max_err` ≥ 0
- `diff_target`, `diff_nontarget` ∈ [0, 1]
- `safety_score` ∈ {"SAFE", "MODERATE", "RISKY", "DANGEROUS"}
- `status` ∈ {"complete", "partial", "negative", "failed"}

## Artifacts

- `experiments/m168_standard_benchmark.py`
- `experiments/m168_standard_benchmark.json`
