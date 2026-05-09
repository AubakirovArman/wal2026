# Docs Command Smoke

Date: 2026-05-09

## Purpose

Verify that the fast public documentation commands still run, while long sweep commands are checked for target existence and kept as explicit reviewer commands.

## Results

- Runnable commands: `22`
- Runnable commands passed: `22`
- Exists-only commands: `2`
- Exists-only commands passed: `2`

## Commands

- Commands with embedded blocked result status: `7`

| Mode | Command | Command Status | Result Status |
|------|---------|----------------|---------------|
| `doc` | `doc exists: README.md` | `PASS` | `—` |
| `doc` | `doc exists: TECHNICAL_REPORT.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/demo_playbook.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/controlled_runners.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/model_small_protocol.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/cross_model_validation_plan.md` | `PASS` | `—` |
| `doc` | `doc exists: docs/robustness_data_protocol.md` | `PASS` | `—` |
| `run` | `PYTHONPATH=src python -m pytest -q tests` | `PASS` | `—` |
| `run` | `PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid` | `PASS` | `—` |
| `run` | `python experiments/m626_technical_report.py` | `PASS` | `PASS` |
| `run` | `python experiments/m627_polished_demo_playbook.py` | `PASS` | `PASS` |
| `run` | `python experiments/m628_blocked_script_taxonomy.py` | `PASS` | `PASS` |
| `run` | `python experiments/m629_controlled_runner_matrix.py` | `PASS` | `PASS` |
| `run` | `python experiments/m630_public_claim_checker.py` | `PASS` | `PASS` |
| `run` | `python experiments/m632_llama_1b_full_workflow.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m633_qwen_small_full_workflow.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m634_gemma_small_full_workflow.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m635_tinyllama_mistral_full_workflow.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m636_cross_model_recipe_replay.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m637_cross_model_layer_aperture.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m638_cross_model_ci_behavior.py` | `PASS` | `BLOCKED` |
| `run` | `python experiments/m639_dirty_facts_corpus.py` | `PASS` | `PASS` |
| `run` | `python experiments/m640_ambiguous_facts_test.py` | `PASS` | `PASS` |
| `run` | `python experiments/m641_temporal_facts_date_logic.py` | `PASS` | `PASS` |
| `run` | `python experiments/m642_long_answer_facts.py` | `PASS` | `PASS` |
| `run` | `python experiments/m643_procedural_knowledge_routing.py` | `PASS` | `PASS` |
| `run` | `python experiments/m644_policy_refusal_edits.py` | `PASS` | `PASS` |
| `run` | `python experiments/m645_hard_facts_hybrid_backend.py` | `PASS` | `SIMULATED` |
| `run` | `python wal_studio_v01/demo.py` | `PASS` | `—` |
| `exists_only` | `python experiments/m624_full_test_inventory.py` | `PASS` | `—` |
| `exists_only` | `python experiments/m625_safe_runtime_sweep.py --timeout 15` | `PASS` | `—` |
