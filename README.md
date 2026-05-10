# WAL — WeightOps Research Framework

![Modules](https://img.shields.io/badge/modules-600+-blue)
![Experiments](https://img.shields.io/badge/experiments-770-blue)
![Status](https://img.shields.io/badge/status-research--prototype-orange)

## Overview

WAL is a research prototype for representing neural-network weights as structured programs.
The repository contains two related tracks:

- **WAL core** — atom/coeff program formats, v1/v2 encoders, decoders, binary serialization, PyTorch integration.
- **DRL/runtime experiments** — route-codebook and Block-RVQ runtime layers for large LLM linear weights.
- **Experiment chronicle** — reproducible milestone scripts, generated result JSON, and a book-style development log.

Current public status: **pre-alpha, fully instrumented research prototype**. See `TECHNICAL_REPORT.md`, `docs/demo_playbook.md`, `KNOWN_ISSUES.md`, and `WAL_AIGI_FULL_CONTEXT.md` for the technical framing, demo path, limitations, release gates, and one-file handoff context.

## Stats

- 796 milestone experiment scripts in `experiments/m*.py`
- 818 Python scripts total in `experiments/`
- 486 experiment result JSON files in `experiments/`
- 631 book/diary entries in `book/`
- 274 top-level result files in `results/`
- 235 docs files in `docs/`
- 5460+ lines in `docs/dev_diary_ru.md`
- 92 Python source modules in `src/`
- 35 pytest tests for the packaged core, audit helpers, and AIGI SDK

## Repository Map

```text
experiments/        milestone scripts and result JSON for M1-M693+
book/               markdown entries for modules, phases, and milestones
docs/               architecture notes, decisions, diaries, and roadmap files
wal_studio_v01/     12-step WAL Studio demo
src/                packaged WAL runtime, v1/v2 APIs, and build utilities
framework/          legacy CLI/framework modules
tests/              unit tests for packaged core APIs
examples/           minimal runnable examples
paper/              article materials, figures, tables, and Russian sections
scripts/            helper scripts for phase workflows and reports
reproduce/          reproduction entry points
results/            legacy summarized result files
archive/            historical generated artifacts, not current release claims
logs/               small text logs for AIGI/WAL audit runs
```

## AIGI Layer

`src/aigi/` starts a separate pre-alpha AIGI SDK layer above WAL. It implements a verified memory loop: propose memory, select tier, verify gates, commit, reject, rollback, and log the result. M679-M693 validate 100 synthetic memory updates, 20 bad-memory rejections, tier routing, rollback, behavioral contracts, experience-to-memory extraction, 25 verified feedback episodes, memory budgets, risk ledger, regression suite, and commit decision reports, and a real Qwen2.5-0.5B-Instruct HF inference backend gate. This is not an AGI claim and does not attach a real semantic weight-edit backend yet.

## Quick Start

```bash
pip install -e .[dev]
python -m wal --help
pytest -q tests
```

CLI surfaces:

```bash
python -m wal core --help
python -m wal studio --help
```

Quickstart example:

```bash
python -m wal studio init local-demo-model
python -m wal studio edit add examples/quickstart/facts.json
python -m wal studio status
```

Demo workflow:

```bash
python wal_studio_v01/demo.py
```

Technical framing:

```text
TECHNICAL_REPORT.md
docs/demo_playbook.md
docs/VALIDATION_STATUS.md
```

Run a single milestone:

```bash
python experiments/m401_memory_leak_fix.py
```

Result validation:

```bash
wal validate-results experiments --fail-on-invalid
python experiments/m510_naming_convention_check.py
python experiments/m518_automated_test_suite.py
python experiments/m544_result_validation.py
python experiments/m624_full_test_inventory.py
python experiments/m625_safe_runtime_sweep.py --timeout 15
python experiments/m626_technical_report.py
python experiments/m627_polished_demo_playbook.py
python experiments/m628_blocked_script_taxonomy.py
python experiments/m629_controlled_runner_matrix.py
python experiments/m630_public_claim_checker.py
python experiments/m631_docs_command_smoke.py
python experiments/m632_llama_1b_full_workflow.py
python experiments/m633_qwen_small_full_workflow.py
python experiments/m634_gemma_small_full_workflow.py
python experiments/m635_tinyllama_mistral_full_workflow.py
python experiments/m636_cross_model_recipe_replay.py
python experiments/m637_cross_model_layer_aperture.py
python experiments/m638_cross_model_ci_behavior.py
python experiments/m639_dirty_facts_corpus.py
python experiments/m640_ambiguous_facts_test.py
python experiments/m641_temporal_facts_date_logic.py
python experiments/m642_long_answer_facts.py
python experiments/m643_procedural_knowledge_routing.py
python experiments/m644_policy_refusal_edits.py
python experiments/m645_hard_facts_hybrid_backend.py
python experiments/m646_negative_test_expansion.py
python experiments/m647_lure_test_expansion.py
python experiments/m648_context_stress_8k_32k.py
python experiments/m649_auto_test_quality_audit.py
python experiments/m650_ci_score_calibration.py
python experiments/m651_behavioral_checksum_drift.py
python experiments/m652_recipe_secret_scanner.py
python experiments/m653_malicious_recipe_injection.py
python experiments/m654_registry_poisoning_test.py
python experiments/m655_hotfix_abuse_test.py
python experiments/m656_prompt_injection_retrieval_context.py
python experiments/m657_provenance_tamper_test.py
python experiments/m658_signed_package_verification.py
python experiments/m659_shadow_deploy_real_server.py
python experiments/m660_canary_real_traffic_simulation.py
python experiments/m661_live_patch_consistency.py
python experiments/m662_emergency_stop_during_build.py
python experiments/m663_emergency_stop_during_inference.py
python experiments/m664_rollback_under_load.py
python experiments/m665_hotfix_with_audit_trail.py
python experiments/m666_24h_soak_test.py
python experiments/m667_memory_leak_long_run.py
python experiments/m668_log_volume_storage_growth.py
python experiments/m669_cli_ux_test.py
python experiments/m670_error_message_quality.py
python experiments/m671_readme_claim_checker.py
python experiments/m672_docs_to_code_consistency.py
python experiments/m673_demo_script_e2e.py
python experiments/m674_github_pages_build.py
python experiments/m675_public_release_dry_run.py
python experiments/m676_public_repo_hardening.py
python experiments/m677_experiment_manifest.py
python experiments/m678_legacy_audit_m1_m50.py
python experiments/m679_aigi_sdk_skeleton.py
python experiments/m680_aigi_100_fact_learning_loop.py
python experiments/m681_aigi_bad_memory_rejection_suite.py
python experiments/m682_aigi_memory_tier_routing.py
python experiments/m683_aigi_rollback_mvp.py
python experiments/m684_aigi_behavioral_contracts.py
python experiments/m685_aigi_experience_to_memory.py
python experiments/m686_aigi_verified_feedback_loop.py
python experiments/m687_aigi_contract_gated_rollback.py
python experiments/m688_single_file_context_digest.py
python experiments/m689_aigi_memory_change_budget.py
python experiments/m690_aigi_risk_ledger.py
python experiments/m691_aigi_contract_regression_suite.py
python experiments/m692_aigi_commit_decision_report.py

# Controlled real-model gate; downloads/loads a small HF model.
python experiments/m693_aigi_real_hf_backend_gate.py
```

## Artifact Policy

Large/generated assets are intentionally excluded from git: model weights, checkpoints, HF caches, `.wal*` workspaces, and binary data artifacts. Keep these in external storage or local cache paths.

## License

MIT
