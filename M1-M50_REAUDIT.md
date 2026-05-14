# M1–M50 Legacy Re-Audit Report

**Audit program**: WAL Legacy Audit v1  
**Batch**: M1–M50 (Early Weight / Route / Encode Probes)  
**Date**: 2026-05-10  
**Method**: Static analysis (imports, CUDA, paths, seed, metrics, destructive ops)

---

## Summary

| Metric | Count |
|--------|-------|
| Total scripts | 55 |
| Runnable (safe) | 4 |
| BLOCKED_GPU | 53 |
| INVALID_BUG (FP16) | 0 |
| NEEDS_RERUN | 0 |
| STILL_VALID | 0 |
| DOC_ONLY | 0 |
| With seed | 17 |
| With result schema | 0 |
| With negative tests | 0 |
| With behavioral checksum | 0 |

---

## Per-Script Analysis

| ID | Category | Lines | Runnable | Status | Seed | Schema | NegTest | BehCheck | Risk Flags | Notes |
|----|----------|-------|----------|--------|------|--------|---------|----------|------------|-------|
| m1_probe_mlp_up | early_weight_probe | 104 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, local_path, no_seed, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Hardcoded local path — not portable |
| m2_codebook_stats | early_weight_probe | 97 | False | BLOCKED_MODEL | ✗ | ✗ | ✗ | ✗ | cuda, local_path, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Hardcoded local path — not portable; Destructive file op detected |
| m3_runtime_bench | early_weight_probe | 108 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, local_path, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Hardcoded local path — not portable; Destructive file op detected |
| m6_route_distill_pilot | early_route | 92 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m20_fast_recon_microbench | early_weight_probe | 110 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Destructive file op detected |
| m21_stage_drop_microbench | meta_release | 99 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; MSE-only metric — weak conclusion; Destructive file op detected |
| m23_id_influence_grammar | early_misc | 293 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m24_per_layer_stage_calibration | meta_release | 174 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m25_same_encoding_runtime_compare | early_weight_probe | 279 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m26_b2_narrow_gate | early_misc | 196 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m26_b3_narrow_gate | early_misc | 206 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m27_fgrl_reencode | early_encode | 439 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m27_ptdp_collect | early_misc | 264 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m27_rrf_collect | early_misc | 319 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m27_rrf_step1a_offline | early_misc | 156 | True | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m27_wal_asm_proto | early_misc | 872 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_cda_proto | early_misc | 903 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, no_result_schema | Uses CUDA — requires GPU rerun |
| m27_wal_dr_proto | early_misc | 1078 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | model_load, exact_only, no_result_schema | Exact-only survival — insufficient under modern CI |
| m27_wal_e2e_proto | early_misc | 1438 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_fg_proto | early_misc | 672 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_hp_proto | early_misc | 637 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m27_wal_ldi_proto | early_misc | 1064 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_lha_proto | early_misc | 1230 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, no_result_schema | Uses CUDA — requires GPU rerun |
| m27_wal_lo_proto | early_misc | 1127 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_lrt_proto | early_misc | 579 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m27_wal_sbc_budget_profile | early_misc | 119 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, mse_only, no_result_schema | No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |
| m27_wal_sbc_core | early_misc | 119 | True | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | no_seed, mse_only, no_result_schema | No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |
| m27_wal_sbc_offline | early_misc | 342 | True | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | no_seed, mse_only, no_result_schema | No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |
| m27_wal_sbc_proto | early_misc | 443 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, no_result_schema | Uses CUDA — requires GPU rerun |
| m27_wal_sbc_tune_proto | early_misc | 146 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m27_wal_ss_proto | early_misc | 575 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m27_wal_ts_proto | early_misc | 916 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, model_load, exact_only, no_result_schema | Uses CUDA — requires GPU rerun; Exact-only survival — insufficient under modern CI |
| m30_path_a_diagnostic | early_misc | 331 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, no_result_schema | Uses CUDA — requires GPU rerun |
| m31_sparse_probe | early_weight_probe | 150 | False | BLOCKED_MODEL | ✗ | ✗ | ✗ | ✗ | no_seed, no_result_schema, destructive_op | No seed + no result schema — nondeterministic, legacy harness; Destructive file op detected |
| m32_path_b_tile_local | early_misc | 140 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; MSE-only metric — weak conclusion |
| m33_encoder_program_cost | early_encode | 261 | True | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | mse_only, no_result_schema, destructive_op | MSE-only metric — weak conclusion; Destructive file op detected |
| m34_m35_m36_encoder_redesign | early_encode | 328 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Destructive file op detected |
| m37_entropy_regularized_encoder | early_encode | 201 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Destructive file op detected |
| m38_vector_route_encoder | early_route | 282 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Destructive file op detected |
| m39_hybrid_encoder | early_encode | 248 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, no_seed, mse_only, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion; Destructive file op detected |
| m40_end_to_end_ppl | early_misc | 228 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema, destructive_op | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; Destructive file op detected |
| m41_load_70b | early_misc | 55 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m42_baseline_ppl_70b | early_misc | 58 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m43_encode_70b | early_encode | 106 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m44_baseline_16steps | early_misc | 79 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema, destructive_op | No seed + no result schema — nondeterministic, legacy harness; Destructive file op detected |
| m44_full_wikitext2_baseline | early_misc | 105 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema, destructive_op | No seed + no result schema — nondeterministic, legacy harness; Destructive file op detected |
| m45_wal_scalar_proto | early_misc | 141 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, mse_only, no_result_schema | No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |
| m46_test_load | early_misc | 21 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m46_wal_scalar_70b_e2e | early_misc | 169 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m46_wal_scalar_70b_e2e_v2 | early_misc | 188 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness |
| m46_wal_scalar_70b_e2e_v3 | early_misc | 173 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | model_load, no_seed, no_result_schema | No seed + no result schema — nondeterministic, legacy harness |
| m47_wal_runtime_test | early_weight_probe | 163 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; MSE-only metric — weak conclusion |
| m48_wal_roundtrip_70b_layer | early_misc | 138 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |
| m49_wal1_vector_atoms | early_misc | 222 | False | BLOCKED_GPU | ✓ | ✗ | ✗ | ✗ | cuda, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; MSE-only metric — weak conclusion |
| m50_wal1_svd_atoms | early_misc | 187 | False | BLOCKED_GPU | ✗ | ✗ | ✗ | ✗ | cuda, model_load, no_seed, mse_only, no_result_schema | Uses CUDA — requires GPU rerun; No seed + no result schema — nondeterministic, legacy harness; MSE-only metric — weak conclusion |

---

## Key Findings

### 1. Zero negative tests in early batch
Ни один из M1–M50 не содержит negative/context/lure тестов. Это ожидаемо для ранних weight probes, но означает, что любые выводы про "сохранение поведения" — слабые.

### 2. Zero behavioral checksums
Ранние эксперименты использовали MSE / exact match, но не behavioral checksum. Под современными правилами это недостаточно.

### 3. Минимум result schema
Большинство M1–M50 не используют `wal.results.v1` schema. Это legacy harness — их результаты нельзя автоматически валидировать.

### 4. Все GPU-зависимые
310 скриптов в полном инвентаре заблокированы по CUDA. M1–M50 — типичные представители: они загружают модели и делают inference на GPU.

### 5. Нет FP16 training в M1–M50
В этом батче нет LoRA/factual editing, поэтому FP16 bug не проявляется. Но и нет seed discipline.

---

## Recommended Modern Reruns

| Priority | Script | Why |
|----------|--------|-----|
| P1 | m4a_full_model_encode.py | Core encode formula — needs seed + PPL gate |
| P1 | m4b_ppl_gate.py | PPL metric itself — compare with modern gate |
| P2 | m6_route_distill_pilot.py | Route distillation — needs modern harness |
| P2 | m27_fgrl_reencode.py | Re-encode logic — check with merge audit |
| P3 | m37_entropy_regularized_encoder.py | Encoder design — validate under fixed seed |

---

## Next Steps

1. **Batch 2**: M51–M100 (LoRA / factual edit / WAL v1) — там ожидается FP16 risk.
2. **Modern harness**: создать `wal audit rerun Mxxx --modernize` CLI.
3. **Critical resurrection**: merge/reencode (M127–M216) после Batch 3.

---

*Generated automatically. Manual review recommended for scripts marked NEEDS_RERUN.*
