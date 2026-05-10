# Legacy Audit M1-M50

Date: 2026-05-10

This is the first Legacy Experiment Resurrection batch. It audits scripts with numeric prefixes M1-M50 using modern M624/M625 safety metadata, source signals, and historical artifact discovery.

## Summary

- Scripts audited: `143`
- Current public claim allowed after audit: `0`
- With historical artifacts: `90`
- With schema-v1 artifacts: `0`

## Review Status Counts

- `blocked_needs_controlled_model_runner`: `133`
- `blocked_needs_slow_runner`: `3`
- `still_valid_needs_schema_v1`: `7`

## Runner Type Counts

- `gpu_or_model_controlled`: `133`
- `safe_core`: `2`
- `safe_core_with_artifact`: `5`
- `slow_safe`: `3`

## Highest Priority Modernization Items

- `move_to_controlled_gpu_or_model_runner`: `133`
- `check_fp32_adapter_or_overflow_control`: `115`
- `record_hardware_requirements`: `101`
- `write_wal_results_v1`: `75`
- `parameterize_model_path_and_record_artifact_hash`: `54`
- `reproduce_or_mark_no_data`: `53`
- `add_explicit_seed`: `51`
- `move_to_slow_runner_with_timeout_budget`: `3`

## Per-Script Review

| File | Runner | Review Status | Artifacts | Key Fixes |
|------|--------|---------------|-----------|-----------|
| `m1_probe_mlp_up.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements, write_wal_results_v1 |
| `m1b_probe_rownorm.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m1c_calibration_sweep.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m2_codebook_stats.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m3_runtime_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m4a_full_model_encode.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m4b_ppl_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m4c_humaneval_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m5a_route_frequency.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6_route_distill_pilot.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `3` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6b_route_distill_sweep.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6c_route_distill_layer_suite.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6d_route_distill_depth_sweep.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6e_local_palette_kernel_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6f_selective_runtime_policy.py` | `safe_core_with_artifact` | `still_valid_needs_schema_v1` | `2` | write_wal_results_v1 |
| `m6g_full_layer_tiled_runtime_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m6h_grouped_local_runtime_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m6i_grouped_2d_runtime_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m6j_grouped_shape_frontier_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m6o_palette_hotness_profile.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6p_hotprefix_frontier_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m6s_shape_runtime_policy.py` | `safe_core_with_artifact` | `still_valid_needs_schema_v1` | `7` | check_fp32_adapter_or_overflow_control, write_wal_results_v1 |
| `m6t_selective_runtime_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `12` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m6u_fused_promotion_policy.py` | `safe_core_with_artifact` | `still_valid_needs_schema_v1` | `2` | write_wal_results_v1 |
| `m6v_baseline_vs_deployment_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m7a_fused_diag.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m7a_fused_real_diag.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m7a_fused_realweight_diag.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m7b_runtime_speed_bench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m7c_threeway_compare.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `3` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m8a_fp8_microbench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m8a_fp8_v2_microbench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m9a_row_archetype_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m9b_codebook_cap_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m9c_act_sparsity_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m10a_block_rvq_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m10b_projection_family_scan.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m10c_block_rvq_global_eval.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `6` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m12a_pq_lowrank_overlay_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m13a_shared_codebook_graph_probe.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m20_fast_recon_microbench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m21_stage_drop_microbench.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m23_id_influence_grammar.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m24_per_layer_stage_calibration.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, reproduce_or_mark_no_data |
| `m25_same_encoding_runtime_compare.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m26_b2_narrow_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m26_b3_narrow_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash |
| `m27_fgrl_reencode.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `3` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_ptdp_collect.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, write_wal_results_v1 |
| `m27_rrf_collect.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_rrf_step1a_offline.py` | `safe_core_with_artifact` | `still_valid_needs_schema_v1` | `2` | write_wal_results_v1 |
| `m27_wal_asm_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_cda_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m27_wal_dr_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | move_to_controlled_gpu_or_model_runner, write_wal_results_v1 |
| `m27_wal_e2e_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_fg_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_hp_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `4` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_ldi_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_lha_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m27_wal_lo_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m27_wal_lrt_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_sbc_budget_profile.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | move_to_controlled_gpu_or_model_runner, write_wal_results_v1 |
| `m27_wal_sbc_core.py` | `safe_core` | `still_valid_needs_schema_v1` | `0` | reproduce_or_mark_no_data |
| `m27_wal_sbc_offline.py` | `safe_core` | `still_valid_needs_schema_v1` | `0` | check_fp32_adapter_or_overflow_control, reproduce_or_mark_no_data |
| `m27_wal_sbc_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m27_wal_sbc_tune_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data, write_wal_results_v1 |
| `m27_wal_ss_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `3` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m27_wal_ts_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, parameterize_model_path_and_record_artifact_hash, record_hardware_requirements |
| `m30_path_a_diagnostic.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m31_sparse_probe.py` | `slow_safe` | `blocked_needs_slow_runner` | `4` | move_to_slow_runner_with_timeout_budget, write_wal_results_v1 |
| `m32_path_b_tile_local.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m33_encoder_program_cost.py` | `safe_core_with_artifact` | `still_valid_needs_schema_v1` | `1` | write_wal_results_v1 |
| `m34_m35_m36_encoder_redesign.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `4` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m37_entropy_regularized_encoder.py` | `slow_safe` | `blocked_needs_slow_runner` | `3` | move_to_slow_runner_with_timeout_budget, record_hardware_requirements, write_wal_results_v1 |
| `m38_vector_route_encoder.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `3` | add_explicit_seed, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, write_wal_results_v1 |
| `m39_hybrid_encoder.py` | `slow_safe` | `blocked_needs_slow_runner` | `3` | add_explicit_seed, move_to_slow_runner_with_timeout_budget, record_hardware_requirements, write_wal_results_v1 |
| `m40_end_to_end_ppl.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m40b_exclude_embeddings.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m40c_higher_quality.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m41_load_70b.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m41b_forward_70b.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m41c_inspect_params.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m41d_load_on_gpus_2_3.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m42_baseline_ppl_70b.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43_encode_70b.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements, reproduce_or_mark_no_data |
| `m43b_analyze_layers.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43c_encode_70b_fast.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43d_encode_70b_batched.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43e_debug_encoder.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43f_debug_vre.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43g_check_dtypes.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43h_check_all_params.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43i_scalar_only.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43j_vre_only.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43k_check_and_ppl.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43l_vre_gate_proj.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43m_vre_all_spiky.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43n_block_mean.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43o_vre_k_proj.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43p_check_devices.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43q_list_spiky.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43r_vre_layer0.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43s_vre_l8_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43t_spiky_threshold.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43u_hybrid_threshold_003.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43v_scalar_l2_q.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43w_scalar_l3_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43x_vre_output_norm.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43y_scalar_k256.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43z_scalar_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43za_scalar_l8_gate.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zb_scalar_l3_v.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zc_scalar_lmax12.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zd_late_layers.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43ze_early_layers.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zf_early_od.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zg_early_smooth.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zh_hybrid_vre_early.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zi_vre_layer0_all.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zj_skip_layer0.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `0` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, reproduce_or_mark_no_data |
| `m43zk_vre_layer0_selective.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zl_scalar_skip_early_spiky.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zm_scalar_lmax10_k256.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zo_scalar_topk_no_lloydmax.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zp_scalar_lmax10_k512.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zq_scalar_lmax10_k1024.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zr_scalar_lmax10_k1024_all_layers.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zs_scalar_lmax10_k2048.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zt_scalar_lmax10_k2048_20steps.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m43zu_baseline_20steps.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner |
| `m43zv_compression_ratio.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner |
| `m43zw_compression_sweep.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner |
| `m44_baseline_16steps.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, write_wal_results_v1 |
| `m44_full_wikitext2_baseline.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `2` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, write_wal_results_v1 |
| `m45_wal_scalar_proto.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner |
| `m46_test_load.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m46_wal_scalar_70b_e2e.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `5` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m46_wal_scalar_70b_e2e_v2.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m46_wal_scalar_70b_e2e_v3.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner |
| `m47_wal_runtime_test.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m48_wal_roundtrip_70b_layer.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m49_wal1_vector_atoms.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | move_to_controlled_gpu_or_model_runner, record_hardware_requirements |
| `m50_wal1_svd_atoms.py` | `gpu_or_model_controlled` | `blocked_needs_controlled_model_runner` | `1` | add_explicit_seed, check_fp32_adapter_or_overflow_control, move_to_controlled_gpu_or_model_runner, record_hardware_requirements |

## Interpretation

M1-M50 are mostly core WAL encoding/runtime experiments. Safe-pass scripts are not upgraded to current public claims until they emit schema-v1 results and clearer hardware/model metadata.
