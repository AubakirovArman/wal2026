# WAL — WeightOps Research Framework

![Modules](https://img.shields.io/badge/modules-600+-blue)
![Experiments](https://img.shields.io/badge/experiments-745-blue)
![Status](https://img.shields.io/badge/status-research--prototype-orange)

## Overview

WAL is a research prototype for representing neural-network weights as structured programs.
The repository contains two related tracks:

- **WAL core** — atom/coeff program formats, v1/v2 encoders, decoders, binary serialization, PyTorch integration.
- **DRL/runtime experiments** — route-codebook and Block-RVQ runtime layers for large LLM linear weights.
- **Experiment chronicle** — reproducible milestone scripts, generated result JSON, and a book-style development log.

## Stats

- 723 milestone experiment scripts in `experiments/m*.py`
- 745 Python scripts total in `experiments/`
- 417 experiment result JSON files in `experiments/`
- 558 book/diary entries in `book/`
- 274 top-level result files in `results/`
- 215 docs files in `docs/`
- 4703 lines in `docs/dev_diary_ru.md`
- 65 Python source modules in `src/`
- 9 pytest tests for the packaged core

## Repository Map

```text
experiments/        milestone scripts and result JSON for M1-M620+
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

Run a single milestone:

```bash
python experiments/m401_memory_leak_fix.py
```

Run a milestone range:

```bash
for i in $(seq 401 410); do
  py_file=$(ls experiments/m${i}_*.py 2>/dev/null | head -1)
  if [ -n "$py_file" ]; then
    echo "=== $(basename "$py_file") ==="
    python "$py_file"
  fi
done
```

Core validation:

```bash
python experiments/m391_final_health_check.py
python experiments/m400_final_system_test.py
python experiments/m412_final_integration_test.py
```

## Artifact Policy

Large/generated assets are intentionally excluded from git: model weights, checkpoints, HF caches, `.wal*` workspaces, and binary data artifacts. Keep these in external storage or local cache paths.

## License

MIT
