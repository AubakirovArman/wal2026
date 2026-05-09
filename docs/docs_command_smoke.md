# Docs Command Smoke

Date: 2026-05-09

## Purpose

Verify that the fast public documentation commands still run, while long sweep commands are checked for target existence and kept as explicit reviewer commands.

## Results

- Runnable commands: `8`
- Runnable commands passed: `8`
- Exists-only commands: `2`
- Exists-only commands passed: `2`

## Commands

| Mode | Command | Status |
|------|---------|--------|
| `doc` | `doc exists: README.md` | `PASS` |
| `doc` | `doc exists: TECHNICAL_REPORT.md` | `PASS` |
| `doc` | `doc exists: docs/demo_playbook.md` | `PASS` |
| `doc` | `doc exists: docs/controlled_runners.md` | `PASS` |
| `run` | `PYTHONPATH=src python -m pytest -q tests` | `PASS` |
| `run` | `PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid` | `PASS` |
| `run` | `python experiments/m626_technical_report.py` | `PASS` |
| `run` | `python experiments/m627_polished_demo_playbook.py` | `PASS` |
| `run` | `python experiments/m628_blocked_script_taxonomy.py` | `PASS` |
| `run` | `python experiments/m629_controlled_runner_matrix.py` | `PASS` |
| `run` | `python experiments/m630_public_claim_checker.py` | `PASS` |
| `run` | `python wal_studio_v01/demo.py` | `PASS` |
| `exists_only` | `python experiments/m624_full_test_inventory.py` | `PASS` |
| `exists_only` | `python experiments/m625_safe_runtime_sweep.py --timeout 15` | `PASS` |
