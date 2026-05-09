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

Current public status: **pre-alpha, fully instrumented research prototype**. See `TECHNICAL_REPORT.md`, `docs/demo_playbook.md`, and `KNOWN_ISSUES.md` for the technical framing, demo path, limitations, and release gates.

## Stats

- 748 milestone experiment scripts in `experiments/m*.py`
- 770 Python scripts total in `experiments/`
- 438 experiment result JSON files in `experiments/`
- 583 book/diary entries in `book/`
- 274 top-level result files in `results/`
- 224 docs files in `docs/`
- 4750+ lines in `docs/dev_diary_ru.md`
- 68 Python source modules in `src/`
- 12 pytest tests for the packaged core

## Repository Map

```text
experiments/        milestone scripts and result JSON for M1-M645+
book/               markdown entries for modules, phases, and milestones
docs/               architecture notes, decisions, diaries, and roadmap files
wal_studio_v01/     12-step WAL Studio demo
src/                packaged WAL runtime, v1/v2 APIs, and build utilities
framework/          legacy CLI/framework modules
tests/              unit tests for packaged core APIs
paper/              article materials, figures, tables, and Russian sections
scripts/            helper scripts for phase workflows and reports
reproduce/          reproduction entry points
results/            legacy summarized result files
```

## Quick Start

```bash
pip install -e .[dev]
python -m wal --help
pytest -q tests
```

Demo workflow:

```bash
python wal_studio_v01/demo.py
```

Technical framing:

```text
TECHNICAL_REPORT.md
docs/demo_playbook.md
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
```

## Artifact Policy

Large/generated assets are intentionally excluded from git: model weights, checkpoints, HF caches, `.wal*` workspaces, and binary data artifacts. Keep these in external storage or local cache paths.

## License

MIT
