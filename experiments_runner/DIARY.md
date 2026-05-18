# Developer Diary — WAL Experiments Runner

Start: 2026-05-16 | GPU: NVIDIA H200 × 8 (IDs 2,3) | venv: .venv/bin/python

## Setup
- Installed WAL + PyTorch 2.12 + CUDA 13.0 in .venv
- Fixed cuda:0/cuda:2 → cuda:3 in all runnable experiments
- Fixed duplicate PYBIND11_MODULE in M30 (removed from _CPP_SOURCE)
- Fixed JSON tensor serialization in M39 (w_hat → w_hat_shape)
- permissions: bypassPermissions

## Completed GPU Experiments

| # | Script | Result | Key finding |
|---|--------|--------|-------------|
| M3 | m3_runtime_bench | PASS | Route encoding 0.12x-0.83x vs dense |
| M8a | m8a_fp8_microbench | PASS | FP8 0.50x storage, 1.20x faster on large mats |
| M20 | m20_fast_recon_microbench | PASS | Block-RVQ: per_group_fast=23ms, rel_mse=4% |
| M21 | m21_stage_drop_microbench | PASS | 3 stages=4%, 2 stages=11%, 1 stage=34% |
| M30 | m30_path_a_diagnostic | PASS | CUDA SMEM kernel compiled, Triton better at low M |
| M31 | m31_sparse_probe | PASS | Codebook usage: top1=0.01-0.05, top8=0.05-0.20 |
| M32 | m32_path_b_tile_local | PASS | Tile-local palette: palette K=512 works best |
| M33 | m33_encoder_program_cost | PASS | Baseline relMSE=0.011, tile_majority=4.84 |
| M34-36 | m34_m35_m36_encoder_redesign | PASS | Blockwise/m35 entropy/m36 non-greedy tested |
| M37 | m37_entropy_regularized_encoder | PASS | K=256: relMSE=0.0047, K=64: relMSE=0.125 |
| M38 | m38_vector_route_encoder | PASS | Vector codebook 64 vecs, relMSE=0.038, bps=1.16 |
| M39 | m39_hybrid_encoder | PASS | Fixed JSON bug, hybrid VRE+scalar works |
| M47 | m47_wal_runtime_test | PASS | Speedup 15.05x, 6/6 tests passed |
| M51 | m51_wal_compiler | PASS | Generic kernel achieves near-peak bandwidth |
| M76 | m76_wal_v1_roundtrip | PASS | 5/5 tests: text↔binary roundtrip |
| M77 | m77_pytorch_integration | PASS | 5/5 tests: device transfer, WALLinear |
| M78 | m78_wal_v1_debugger | PASS | 7/7 tests: breakpoints, heatmaps, state |
| M79 | m79_stdlib_prototype | PASS | 6/6 tests: atom library, queries, transfer |
| M80 | m80_hardware_backends | PASS | 8/8 tests: CPU/CUDA/MPS/ROCm scaffold |
| M91 | m91_qat_differentiable_decode | PASS | 5/5 tests: 1.38× improvement via table-tuning |
| M92 | m92_wal_native_lora | PASS | 3/3 tests: native LoRA merge losslessly |
| M94 | m94_qat_reencode | PASS | 1/1: periodic re-encoding works |
| M95 | m95_qat_full_pipeline | PASS | 1/1: full QAT pipeline works |
| M132 | m132_runtime_bench | PASS | WAL vs dense: 0.92x-1.01x slowdown |
| M152 | m152_safety_score_fast | PASS | SAFE/MODERATE/RISKY/DANGEROUS classification |
| M166 | m166_soft_wallinear_v2 | PASS | WAL final 7.03 vs baseline 7.16 |
| M167 | m167_ste_gumbel_v2 | PASS | Gumbel viable: True |
| M171 | m171_unified_runtime_pipeline | PASS | WALModel API: save/load/merge works |
| M175 | m175_gumbel_scale_up | PASS | Gumbel matches dense at scale |
| M176 | m176_factorized_logits | PASS | Factorized/Dense ratio: 1.00-1.01 |
| M177 | m177_temperature_schedule | PASS | cosine_decay best schedule |
| M192 | m192_gumbel_wave_regularization | PASS | Wave regularization tested |

## Completed CPU Validation

| # | Script | Result |
|---|--------|--------|
| M622 | m622_result_schema_gate | PASS: 490/490 valid |
| M623 | m623_core_release_gate | PASS: 35/35 tests |
| M624 | m624_full_test_inventory | PASS: 289 runnable, 533 blocked |
| M625 | m625_safe_runtime_sweep | FAIL: 1 fail (1 new) |
| M680 | m680_aigi_100_fact_learning_loop | PASS: 100/100 facts |
| M686 | m686_aigi_verified_feedback_loop | PASS: 25/25 episodes |

## Failed / Skipped

| # | Reason |
|---|--------|
| M1, M1b, M1c, M2, M5a, M6, M7a, M7b, M9a, M9b | Require dwl2_dynamic_route package |
| M10a, M10b, M12a | Require dwl2_dynamic_route |
| M49 | OOM: needs 56 GiB allocation |
| M152_structured | Running in background |
| All M100-M250 | Require HF model downloads |
| All M40-M48, M50+ range | Require model snapshots |
