# WAL Studio v0.1

Pre-alpha WeightOps demo for recipe/build/test/debug/rollback workflows.

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

- 800 Python experiment/prototype scripts
- 468 result JSON files
- 613 book entries
- M621-M675 cleanup/report/demo/runner/robustness/CI/security/deployment/product gates

## Validation Results

| Metric | Value |
|--------|-------|
| Core pytest | 12 passing |
| Result schema | valid |
| M624 inventory | 0 compile failures |
| M625 safe sweep | 279 PASS, 0 FAIL, 521 BLOCKED |
| Demo framing | pre-alpha |

## Honest Assessment

**Working**: CLI concepts, recipes, DAG/build concepts, CI gates, rollback, blame, bisect, release notes.
**Needs work**: controlled cross-model validation, real GPU-heavy workflows, non-simulated deployment integrations.
**Status**: Pre-alpha, research-grade prototype.

See `../TECHNICAL_REPORT.md` and `../docs/demo_playbook.md` for current public framing.

## Architecture

```
recipes/ → build → wal_weights.bin
              ↓
         ci_gate/ → pass/fail
              ↓
         deploy → inference
```

## License

MIT
