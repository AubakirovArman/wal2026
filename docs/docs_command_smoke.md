# Docs Command Smoke

Date: 2026-05-10

## Purpose

Verify that the fast public documentation commands still run, while long sweep commands are checked for target existence and kept as explicit reviewer commands.

## Results

- Runnable commands: `53`
- Runnable commands passed: `53`
- Exists-only commands: `2`
- Exists-only commands passed: `2`

## Commands

- Commands with embedded blocked result status: `2`

| Mode | Command | Command Status | Result Status |
|------|---------|----------------|---------------|
| `doc` | `doc exists: README.md` | `PASS` | `—` |
| `doc` | `doc exists: TECHNICAL_REPORT.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/VALIDATION_STATUS.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/project_metrics.json` | `PASS` | `—` |
| `doc` | `doc exists: docs/demo_playbook.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/controlled_runners.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/model_small_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/cross_model_validation_plan.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/robustness_data_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/ci_hardening_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/security_hardening_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/deployment_reality_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/product_polish_protocol.md` | `PASS` | `—` |
| `run` | `PYTHONPATH=src python -m pytest -q tests` | `PASS` | `—` |
| `run` | `PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid` | `PASS` | `—` |
| `run` | `python experiments/m626_technical_report.py` | `PASS` | `PASS` |
| `run` | `python experiments/m627_polished_demo_playbook.py` | `PASS` | `PASS` |
| `run` | `python experiments/m628_blocked_script_taxonomy.py` | `PASS` | `PASS` |
| `run` | `python experiments/m629_controlled_runner_matrix.py` | `PASS` | `PASS` |
| `run` | `python experiments/m630_public_claim_checker.py` | `PASS` | `PASS` |
| `run` | `python experiments/m632_llama_1b_full_workflow.py` | `PASS` | `PASS` |
| `run` | `python experiments/m633_qwen_small_full_workflow.py` | `PASS` | `PASS` |
| `run` | `python experiments/m634_gemma_small_full_workflow.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m635_tinyllama_mistral_full_workflow.py` | `PASS` | `PASS` |
| `run` | `python experiments/m636_cross_model_recipe_replay.py` | `PASS` | `PASS` |
| `run` | `python experiments/m637_cross_model_layer_aperture.py` | `PASS` | `PASS` |
| `run` | `python experiments/m638_cross_model_ci_behavior.py` | `PASS` | `PASS` |
| `run` | `python experiments/m639_dirty_facts_corpus.py` | `PASS` | `PASS` |
| `run` | `python experiments/m640_ambiguous_facts_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m641_temporal_facts_date_logic.py` | `PASS` | `PASS` |
| `run` | `python experiments/m642_long_answer_facts.py` | `PASS` | `PASS` |
| `run` | `python experiments/m643_procedural_knowledge_routing.py` | `PASS` | `PASS` |
| `run` | `python experiments/m644_policy_refusal_edits.py` | `PASS` | `PASS` |
| `run` | `python experiments/m645_hard_facts_hybrid_backend.py` | `PASS` | `SIMULATED` |
| `run` | `python experiments/m646_negative_test_expansion.py` | `PASS` | `PASS` |
| `run` | `python experiments/m647_lure_test_expansion.py` | `PASS` | `PASS` |
| `run` | `python experiments/m648_context_stress_8k_32k.py` | `PASS` | `PASS` |
| `run` | `python experiments/m649_auto_test_quality_audit.py` | `PASS` | `PASS` |
| `run` | `python experiments/m650_ci_score_calibration.py` | `PASS` | `PASS` |
| `run` | `python experiments/m651_behavioral_checksum_drift.py` | `PASS` | `PASS` |
| `run` | `python experiments/m652_recipe_secret_scanner.py` | `PASS` | `PASS` |
| `run` | `python experiments/m653_malicious_recipe_injection.py` | `PASS` | `PASS` |
| `run` | `python experiments/m654_registry_poisoning_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m655_hotfix_abuse_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m656_prompt_injection_retrieval_context.py` | `PASS` | `PASS` |
| `run` | `python experiments/m657_provenance_tamper_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m658_signed_package_verification.py` | `PASS` | `PASS` |
| `run` | `python experiments/m659_shadow_deploy_real_server.py` | `PASS` | `PASS` |
| `run` | `python experiments/m660_canary_real_traffic_simulation.py` | `PASS` | `PASS` |
| `run` | `python experiments/m661_live_patch_consistency.py` | `PASS` | `PASS` |
| `run` | `python experiments/m662_emergency_stop_during_build.py` | `PASS` | `PASS` |
| `run` | `python experiments/m663_emergency_stop_during_inference.py` | `PASS` | `PASS` |
| `run` | `python experiments/m664_rollback_under_load.py` | `PASS` | `PASS` |
| `run` | `python experiments/m665_hotfix_with_audit_trail.py` | `PASS` | `PASS` |
| `run` | `python experiments/m666_24h_soak_test.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m667_memory_leak_long_run.py` | `PASS` | `SIMULATED` |
| `run` | `python experiments/m668_log_volume_storage_growth.py` | `PASS` | `PASS` |
| `run` | `python experiments/m669_cli_ux_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m670_error_message_quality.py` | `PASS` | `PASS` |
| `run` | `python experiments/m671_readme_claim_checker.py` | `PASS` | `PASS` |
| `run` | `python experiments/m672_docs_to_code_consistency.py` | `PASS` | `PASS` |
| `run` | `python experiments/m673_demo_script_e2e.py` | `PASS` | `PASS` |
| `run` | `python experiments/m674_github_pages_build.py` | `PASS` | `PASS` |
| `run` | `python experiments/m675_public_release_dry_run.py` | `PASS` | `PASS` |
| `run` | `python experiments/m676_public_repo_hardening.py` | `PASS` | `PASS` |
| `run` | `python wal_studio_v01/demo.py` | `PASS` | `—` |
| `exists_only` | `python experiments/m624_full_test_inventory.py` | `PASS` | `—` |
| `exists_only` | `python experiments/m625_safe_runtime_sweep.py --timeout 15` | `PASS` | `—` |
