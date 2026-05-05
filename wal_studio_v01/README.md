# WAL Studio v0.1

WeightOps framework for knowledge surgery on LLMs.

## Quick Start

```bash
python wal_studio_v01/demo.py
```

## Core Concepts

- **Recipe** — atomic knowledge edit (template + variables)
- **Build** — compile recipes into WAL weights
- **CI** — continuous integration for edits
- **Blame** — identify which edit caused regression
- **Bisect** — binary search first bad commit

## Project Statistics

- 507 experiments
- 314 book entries
- 95 modules (M291–M385)
- 5 validation experiments (E1–E5)

## Validation Results

| Metric | Value |
|--------|-------|
| Survival (synthetic) | 95.2% |
| Survival (realistic) | 90.4% |
| CI Score | 94% |
| Security | 7/8 |
| 24h Server | 0.85% errors |

## Honest Assessment

**Working**: CLI, recipes, DAG, build, CI, rollback, blame, bisect
**Needs work**: Multi-model validation, prompt injection, memory growth
**Status**: Pre-alpha, research-grade prototype

## Architecture

```
recipes/ → build → wal_weights.bin
              ↓
         ci_gate/ → pass/fail
              ↓
         deploy → inference
```

## License

Research prototype. Not for production without further validation.
