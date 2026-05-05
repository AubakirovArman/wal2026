# Experiment Diary

Chronological notes for all experiments. Each file covers one experiment or a tight batch of related experiments.

## Structure

- One experiment per file (or one tightly related batch)
- Include date, goal, configuration, result, and artifacts
- Link to related experiments
- Note negative results explicitly — they are as valuable as positive ones

## Phase Index

### Phase 1 — Scalar Quantization (M1–M10)

| Experiment | Notes |
|-----------|-------|
| [m10a_block_rvq_probe](m10a_block_rvq_probe.md) | See file for details |
| [m10b_projection_family_scan](m10b_projection_family_scan.md) | See file for details |
| [m10c_block_rvq_global_eval](m10c_block_rvq_global_eval.md) | PPL evaluation. |
| [m1_probe_mlp_up](m1_probe_mlp_up.md) | First probe of MLP up_proj weights. Row-norm statistics coll |
| [m1b_probe_rownorm](m1b_probe_rownorm.md) | Row-norm calibration. Verified that per-row normalization st |
| [m1c_calibration_sweep](m1c_calibration_sweep.md) | Grid search over scale and zero-point parameters. |
| [m2_codebook_stats](m2_codebook_stats.md) | Codebook entropy and diversity analysis. ~1500 unique routes |
| [m3_runtime_bench](m3_runtime_bench.md) | Inference speed microbenchmark. Baseline for runtime compari |
| [m4a_full_model_encode](m4a_full_model_encode.md) | before committing to full-model PPL tests. |
| [m4b_ppl_gate](m4b_ppl_gate.md) | PPL evaluation. |
| [m4c_humaneval_gate](m4c_humaneval_gate.md) | HumanEval gate. limit=20: dense 0.7, routed 0.7. limit=164:  |
| [m5a_route_frequency](m5a_route_frequency.md) | Routing frequency analysis across layers and stages. |
| [m6_route_distill_pilot](m6_route_distill_pilot.md) | See file for details |
| [m6b_route_distill_sweep](m6b_route_distill_sweep.md) | See file for details |
| [m6c_route_distill_layer_suite](m6c_route_distill_layer_suite.md) | See file for details |
| [m6d_route_distill_depth_sweep](m6d_route_distill_depth_sweep.md) | See file for details |
| [m6e_local_palette_kernel_bench](m6e_local_palette_kernel_bench.md) | See file for details |
| [m6f_selective_runtime_policy](m6f_selective_runtime_policy.md) | See file for details |
| [m6g_full_layer_tiled_runtime_bench](m6g_full_layer_tiled_runtime_bench.md) | See file for details |
| [m6h_grouped_local_runtime_bench](m6h_grouped_local_runtime_bench.md) | See file for details |
| [m6i_grouped_2d_runtime_bench](m6i_grouped_2d_runtime_bench.md) | See file for details |
| [m6j_grouped_shape_frontier_bench](m6j_grouped_shape_frontier_bench.md) | See file for details |
| [m6o_palette_hotness_profile](m6o_palette_hotness_profile.md) | See file for details |
| [m6p_hotprefix_frontier_bench](m6p_hotprefix_frontier_bench.md) | See file for details |
| [m6s_shape_runtime_policy](m6s_shape_runtime_policy.md) | See file for details |
| [m6t_selective_runtime_gate](m6t_selective_runtime_gate.md) | PPL evaluation. |
| [m6u_fused_promotion_policy](m6u_fused_promotion_policy.md) | See file for details |
| [m6v_baseline_vs_deployment_gate](m6v_baseline_vs_deployment_gate.md) | PPL evaluation. |
| [m7a_fused_diag](m7a_fused_diag.md) | See file for details |
| [m7a_fused_real_diag](m7a_fused_real_diag.md) | See file for details |
| [m7a_fused_realweight_diag](m7a_fused_realweight_diag.md) | See file for details |
| [m7b_runtime_speed_bench](m7b_runtime_speed_bench.md) | See file for details |
| [m7c_threeway_compare](m7c_threeway_compare.md) | Runs WikiText-2 raw PPL + throughput + VRAM for: |
| [m8a_fp8_microbench](m8a_fp8_microbench.md) | See file for details |
| [m8a_fp8_v2_microbench](m8a_fp8_v2_microbench.md) | See file for details |
| [m9a_row_archetype_probe](m9a_row_archetype_probe.md) | See file for details |
| [m9b_codebook_cap_probe](m9b_codebook_cap_probe.md) | PPL evaluation. |
| [m9c_act_sparsity_probe](m9c_act_sparsity_probe.md) | See file for details |

### Phase 2 — Tile/Vector Encoding (M12–M39)

| Experiment | Notes |
|-----------|-------|
| [m12a_pq_lowrank_overlay_probe](m12a_pq_lowrank_overlay_probe.md) | See file for details |
| [m13a_shared_codebook_graph_probe](m13a_shared_codebook_graph_probe.md) | See file for details |
| [m20_fast_recon_microbench](m20_fast_recon_microbench.md) | Fast reconstruct: caches bf16 codebook representations. |
| [m21_stage_drop_microbench](m21_stage_drop_microbench.md) | - baseline `3` stage: `PPL = 2.9487`, `385.47 tok/s` |
| [m23_id_influence_grammar](m23_id_influence_grammar.md) | Grammar (stage, id) influence analysis. ID vocabulary is rea |
| [m24_per_layer_stage_calibration](m24_per_layer_stage_calibration.md) | Per-layer stage calibration. |
| [m25_same_encoding_runtime_compare](m25_same_encoding_runtime_compare.md) | PPL evaluation. |
| [m26_b2_narrow_gate](m26_b2_narrow_gate.md) | - `full_weight_fast`   PPL `32.2524`, `947.92` tok/s, peak V |
| [m26_b3_narrow_gate](m26_b3_narrow_gate.md) | - `full_weight_fast`   PPL `32.2524`, `947.92` tok/s, peak V |
| [m27_fgrl_reencode](m27_fgrl_reencode.md) | - slice-level eval metrics: PPL, tok/s, peak VRAM on the sam |
| [m27_ptdp_collect](m27_ptdp_collect.md) | See file for details |
| [m27_rrf_collect](m27_rrf_collect.md) | See file for details |
| [m27_rrf_step1a_offline](m27_rrf_step1a_offline.md) | See file for details |
| [m27_wal_asm_proto](m27_wal_asm_proto.md) | PPL evaluation. |
| [m27_wal_cda_proto](m27_wal_cda_proto.md) | PPL evaluation. |
| [m27_wal_dr_proto](m27_wal_dr_proto.md) | PPL evaluation. |
| [m27_wal_e2e_proto](m27_wal_e2e_proto.md) | PPL evaluation. |
| [m27_wal_fg_proto](m27_wal_fg_proto.md) | PPL evaluation. |
| [m27_wal_hp_proto](m27_wal_hp_proto.md) | PPL evaluation. |
| [m27_wal_ldi_proto](m27_wal_ldi_proto.md) | PPL evaluation. |
| [m27_wal_lha_proto](m27_wal_lha_proto.md) | PPL evaluation. |
| [m27_wal_lo_proto](m27_wal_lo_proto.md) | PPL evaluation. |
| [m27_wal_lrt_proto](m27_wal_lrt_proto.md) | PPL evaluation. |
| [m27_wal_sbc_budget_profile](m27_wal_sbc_budget_profile.md) | See file for details |
| [m27_wal_sbc_core](m27_wal_sbc_core.md) | See file for details |
| [m27_wal_sbc_offline](m27_wal_sbc_offline.md) | See file for details |
| [m27_wal_sbc_proto](m27_wal_sbc_proto.md) | PPL evaluation. |
| [m27_wal_sbc_tune_proto](m27_wal_sbc_tune_proto.md) | PPL evaluation. |
| [m27_wal_ss_proto](m27_wal_ss_proto.md) | PPL evaluation. |
| [m27_wal_ts_proto](m27_wal_ts_proto.md) | PPL evaluation. |
| [m30_path_a_diagnostic](m30_path_a_diagnostic.md) | See file for details |
| [m31_sparse_probe](m31_sparse_probe.md) | See file for details |
| [m32_path_b_diagnostic](m32_path_b_diagnostic.md) | See file for details |
| [m32_path_b_tile_local](m32_path_b_tile_local.md) | See file for details |
| [m33_encoder_program_cost](m33_encoder_program_cost.md) | See file for details |
| [m34_m35_m36_encoder_redesign](m34_m35_m36_encoder_redesign.md) | See file for details |
| [m35_cross_layer_analysis](m35_cross_layer_analysis.md) | | Step 5: Integration | Cannot evaluate without PPL access | |
| [m37_entropy_regularized](m37_entropy_regularized.md) | See file for details |
| [m37_entropy_regularized_encoder](m37_entropy_regularized_encoder.md) | See file for details |
| [m38_vector_route_encoder](m38_vector_route_encoder.md) | See file for details |
| [m39_hybrid_encoder](m39_hybrid_encoder.md) | See file for details |
| [m39_hybrid_encoder_final](m39_hybrid_encoder_final.md) | End-to-end PPL on WikiText-2 (2048 tokens): |

### Phase 3 — Full Model Encode (M40–M59)

| Experiment | Notes |
|-----------|-------|
| [m40_end_to_end_ppl](m40_end_to_end_ppl.md) | M40: End-to-end PPL benchmark on Llama 3.1 8B with hybrid en |
| [m40b_exclude_embeddings](m40b_exclude_embeddings.md) | PPL evaluation. |
| [m40c_higher_quality](m40c_higher_quality.md) | PPL evaluation. |
| [m41_load_70b](m41_load_70b.md) | See file for details |
| [m41b_forward_70b](m41b_forward_70b.md) | See file for details |
| [m41c_inspect_params](m41c_inspect_params.md) | See file for details |
| [m41d_load_on_gpus_2_3](m41d_load_on_gpus_2_3.md) | See file for details |
| [m42_baseline_ppl_70b](m42_baseline_ppl_70b.md) | M42: Baseline PPL on Llama 3.3 70B (WikiText-2). |
| [m43_70b_end_to_end_encoding](m43_70b_end_to_end_encoding.md) | - Baseline PPL (WikiText-2, first 6656 tokens, 10 steps): ** |
| [m43_encode_70b](m43_encode_70b.md) | M43: Apply hybrid encoder to Llama 3.3 70B and measure PPL. |
| [m43b_analyze_layers](m43b_analyze_layers.md) | See file for details |
| [m43c_encode_70b_fast](m43c_encode_70b_fast.md) | PPL evaluation. |
| [m43d_encode_70b_batched](m43d_encode_70b_batched.md) | PPL evaluation. |
| [m43e_debug_encoder](m43e_debug_encoder.md) | See file for details |
| [m43f_debug_vre](m43f_debug_vre.md) | See file for details |
| [m43g_check_dtypes](m43g_check_dtypes.md) | See file for details |
| [m43h_check_all_params](m43h_check_all_params.md) | See file for details |
| [m43i_scalar_only](m43i_scalar_only.md) | PPL evaluation. |
| [m43j_vre_only](m43j_vre_only.md) | PPL evaluation. |
| [m43k_check_and_ppl](m43k_check_and_ppl.md) | M43k: Encode all params (same as m43h) and measure PPL. |
| [m43l_vre_gate_proj](m43l_vre_gate_proj.md) | See file for details |
| [m43m_vre_all_spiky](m43m_vre_all_spiky.md) | PPL evaluation. |
| [m43n_block_mean](m43n_block_mean.md) | See file for details |
| [m43o_vre_k_proj](m43o_vre_k_proj.md) | PPL evaluation. |
| [m43p_check_devices](m43p_check_devices.md) | See file for details |
| [m43q_list_spiky](m43q_list_spiky.md) | See file for details |
| [m43r_vre_layer0](m43r_vre_layer0.md) | PPL evaluation. |
| [m43s_vre_l8_gate](m43s_vre_l8_gate.md) | PPL evaluation. |
| [m43t_spiky_threshold](m43t_spiky_threshold.md) | See file for details |
| [m43u_hybrid_threshold_003](m43u_hybrid_threshold_003.md) | PPL evaluation. |
| [m43v_scalar_l2_q](m43v_scalar_l2_q.md) | PPL evaluation. |
| [m43w_scalar_l3_gate](m43w_scalar_l3_gate.md) | PPL evaluation. |
| [m43x_vre_output_norm](m43x_vre_output_norm.md) | See file for details |
| [m43y_scalar_k256](m43y_scalar_k256.md) | PPL evaluation. |
| [m43z_scalar_gate](m43z_scalar_gate.md) | PPL evaluation. |
| [m43za_scalar_l8_gate](m43za_scalar_l8_gate.md) | PPL evaluation. |
| [m43zb_scalar_l3_v](m43zb_scalar_l3_v.md) | PPL evaluation. |
| [m43zc_scalar_lmax12](m43zc_scalar_lmax12.md) | PPL evaluation. |
| [m43zd_late_layers](m43zd_late_layers.md) | PPL evaluation. |
| [m43ze_early_layers](m43ze_early_layers.md) | PPL evaluation. |
| [m43zf_early_od](m43zf_early_od.md) | PPL evaluation. |
| [m43zg_early_smooth](m43zg_early_smooth.md) | PPL evaluation. |
| [m43zh_hybrid_vre_early](m43zh_hybrid_vre_early.md) | PPL evaluation. |
| [m43zi_vre_layer0_all](m43zi_vre_layer0_all.md) | PPL evaluation. |
| [m43zj_skip_layer0](m43zj_skip_layer0.md) | PPL evaluation. |
| [m43zk_vre_layer0_selective](m43zk_vre_layer0_selective.md) | PPL evaluation. |
| [m43zl_scalar_skip_early_spiky](m43zl_scalar_skip_early_spiky.md) | PPL evaluation. |
| [m43zm_scalar_lmax10_k256](m43zm_scalar_lmax10_k256.md) | PPL evaluation. |
| [m43zo_scalar_topk_no_lloydmax](m43zo_scalar_topk_no_lloydmax.md) | PPL evaluation. |
| [m43zp_scalar_lmax10_k512](m43zp_scalar_lmax10_k512.md) | PPL evaluation. |
| [m43zq_scalar_lmax10_k1024](m43zq_scalar_lmax10_k1024.md) | PPL evaluation. |
| [m43zr_scalar_lmax10_k1024_all_layers](m43zr_scalar_lmax10_k1024_all_layers.md) | PPL evaluation. |
| [m43zs_scalar_lmax10_k2048](m43zs_scalar_lmax10_k2048.md) | PPL evaluation. |
| [m43zt_scalar_lmax10_k2048_20steps](m43zt_scalar_lmax10_k2048_20steps.md) | PPL evaluation. |
| [m43zu_baseline_20steps](m43zu_baseline_20steps.md) | M43zu: Baseline PPL on 20 steps for fair comparison. |
| [m43zv_compression_ratio](m43zv_compression_ratio.md) | See file for details |
| [m43zw_compression_sweep](m43zw_compression_sweep.md) | See file for details |
| [m44_baseline_16steps](m44_baseline_16steps.md) | M44: Baseline PPL on 16 steps (quick gate). |
| [m44_full_wikitext2_baseline](m44_full_wikitext2_baseline.md) | PPL evaluation. |
| [m44_wal_language_inception](m44_wal_language_inception.md) | | Config | PPL (10-step) | PPL (20-step) | Notes | |
| [m45_wal_scalar_proto](m45_wal_scalar_proto.md) | 3. Test on full 70B model with PPL gate |
| [m46_test_load](m46_test_load.md) | - **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0 |
| [m46_wal_scalar_70b_e2e](m46_wal_scalar_70b_e2e.md) | WAL scalar full 70B encode + PPL |
| [m46_wal_scalar_70b_e2e_v2](m46_wal_scalar_70b_e2e_v2.md) | PPL evaluation. |
| [m46_wal_scalar_70b_e2e_v3](m46_wal_scalar_70b_e2e_v3.md) | PPL evaluation. |
| [m47_wal_runtime_test](m47_wal_runtime_test.md) | WAL Runtime — decode, round-trip, serialization test. |
| [m48_wal_roundtrip_70b_layer](m48_wal_roundtrip_70b_layer.md) | Round-trip on real layer from 70B model. Correctness verifie |
| [m49_wal1_vector_atoms](m49_wal1_vector_atoms.md) | WAL-1 vector atoms — CATASTROPHIC PPL. Important negative re |
| [m50_wal1_svd_atoms](m50_wal1_svd_atoms.md) | SVD-based atoms — also catastrophic. |
| [m51_wal_compiler](m51_wal_compiler.md) | WAL Compiler — compilation gives no win for K=128. |
| [m52_cross_layer_atom_sharing](m52_cross_layer_atom_sharing.md) | Cross-layer atom sharing beats per-layer atoms. |
| [m53_wal_compression_ppl](m53_wal_compression_ppl.md) | M53: WAL-0 compression + PPL validation on Llama 3.3 70B. |
| [m53b_fused_triton_encode](m53b_fused_triton_encode.md) | See file for details |
| [m53c_wal_fused_encode_ppl](m53c_wal_fused_encode_ppl.md) | M53c: WAL-0 with fused Triton encode + PPL on Llama 3.3 70B. |
| [m54a_wal_codebook_mining](m54a_wal_codebook_mining.md) | See file for details |
| [m54b_wal_codebook_decode](m54b_wal_codebook_decode.md) | See file for details |
| [m55a_wal_variable_length](m55a_wal_variable_length.md) | PPL evaluation. |
| [m56a_wal_grammar_analysis](m56a_wal_grammar_analysis.md) | See file for details |
| [m57_debug_codebook](m57_debug_codebook.md) | - **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0 |
| [m57_debug_codebook_detail](m57_debug_codebook_detail.md) | See file for details |
| [m57_wal_codebook_70b_ppl](m57_wal_codebook_70b_ppl.md) | PPL 2.7828, encode time 437s (7.3 min) — 6× faster than M53c |
| [m58_wal_codec_v2_global](m58_wal_codec_v2_global.md) | apply precomputed recon, measure PPL and compression. |
| [m59_wal_global_codebook_fast](m59_wal_global_codebook_fast.md) | Encode is identical to M57 (per-layer atoms, PPL 2.7828 prov |
| [m59_wal_global_codebook_fast_v3](m59_wal_global_codebook_fast_v3.md) | PPL evaluation. |
| [m59a_wal_global_codebook](m59a_wal_global_codebook.md) | See file for details |

### Phase 4 — WAL v2 (M60–M64)

| Experiment | Notes |
|-----------|-------|
| [m60_wal_v2_scalar_prototype](m60_wal_v2_scalar_prototype.md) | PPL evaluation. |
| [m61_wal_v2_70b_ppl](m61_wal_v2_70b_ppl.md) | | Method | PPL | |
| [m62_wal_v2_grammar_asm](m62_wal_v2_grammar_asm.md) | See file for details |
| [m63_wal_v2_vm_runtime](m63_wal_v2_vm_runtime.md) | See file for details |
| [m64_wal_v2_compression](m64_wal_v2_compression.md) | See file for details |

### Phase 5 — WAL v1 Hierarchical (M65–M77)

| Experiment | Notes |
|-----------|-------|
| [m65_wal_v1_tile_prototype](m65_wal_v1_tile_prototype.md) | Tile/vector quantization prototype. Single-layer OK but full |
| [m66_wal_v1_pq_prototype](m66_wal_v1_pq_prototype.md) | Product Quantization prototype. |
| [m67_pq_systematic](m67_pq_systematic.md) | Two-tier PQ systematic test. 8 bits = DEGRADE (3.1137), 12 b |
| [m68_svd_prototype](m68_svd_prototype.md) | Truncated SVD + quantization. relMSE 0.55-0.99, toxic. |
| [m69_pq_varying_k](m69_pq_varying_k.md) | Position-specific sweep K=16,32,64,128,256. K=16→111k FAIL,  |
| [m70_ppl_position_specific](m70_ppl_position_specific.md) | M70: Full 70B PPL with position-specific scalar quantization |
| [m71_single_layer_ppl_validation](m71_single_layer_ppl_validation.md) | M71: Single-layer PPL validation of M65-M69 findings. |
| [m72_full_ppl_m69_sweep](m72_full_ppl_m69_sweep.md) | M72: Full-model PPL sweep for M69 position-specific quantiza |
| [m73_full_ppl_twotier](m73_full_ppl_twotier.md) | M73: Full-model PPL for two-tier uniform quantization. |
| [m74_wal_v1_two_term_prototype](m74_wal_v1_two_term_prototype.md) | Two-term greedy (32 bits) excellent relMSE but subroutine cl |
| [m75_wal_v1_70b_ppl](m75_wal_v1_70b_ppl.md) | M75: WAL v1 full 70B PPL + round-trip verification. |
| [m76_wal_v1_roundtrip](m76_wal_v1_roundtrip.md) | PPL evaluation. |
| [m77_pytorch_integration](m77_pytorch_integration.md) | 5/5 tests PASS: WALParameter, WALLinear Forward, WALCachedLi |

## Cross-Reference Map

```
M1-M10    → Calibration & baselines
M12-M39   → Runtime prototypes & grammar induction
M40-M59   → Full 70B encode & quality gates
M60-M64   → WAL v2 language & codec
M65-M77   → WAL v1 hierarchical atoms & PyTorch integration
```

## Quality Timeline

| Milestone | PPL | Delta vs Baseline | Notes |
|-----------|-----|-------------------|-------|
| Baseline (M42) | 2.7805 | — | Authoritative |
| Scalar DRL v2 best (M43zj) | 4.26 | +1.48 | Skip layer 0 |
| WAL-0 Codebook (M57) | 2.7828 | +0.0023 | 6× faster encode |
| WAL v2 (M61) | 2.7781 | −0.0024 | Production codec |
| WAL v1 (M75) | 2.7809 | +0.0004 | Hierarchical atoms |

## Rules for Adding Entries

1. One experiment per file
2. Include date, goal, configuration, result, and artifacts
3. Link to related experiments
4. Note negative results explicitly — they are as valuable as positive ones
