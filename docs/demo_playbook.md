# WAL Studio Demo Playbook

Date: 2026-05-09  
Audience: public GitHub demo, technical reviewers, early collaborators  
Mode: pre-alpha product walkthrough with simulated model behavior where needed

## Goal

Show WAL as a WeightOps workflow, not as a finished model product.

The demo should prove one narrow thing:

```text
model edits can be represented as recipes,
built into versioned artifacts,
tested through behavioral gates,
debugged when a regression appears,
and rolled back with traceable release notes.
```

## Demo Contract

Use conservative language during the demo:

- say "pre-alpha framework" instead of stronger release claims;
- say "simulated deployment path" for canary/shadow/hotfix flows;
- distinguish real unit tests from simulated model behavior;
- explain `BLOCKED` and `UNSUPPORTED` as honest statuses, not hidden failures.

## Setup

```bash
cd /mnt/hf_model_weights/arman/3bit/wal
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid
python wal_studio_v01/demo.py
```

Expected gates:

- unit tests pass;
- result JSON schema validation passes;
- WAL Studio demo prints a 12-step walkthrough.

## Main Story

The best public story is a nine-step regression workflow.

| Step | Action | What To Show |
|------|--------|--------------|
| 1 | Init workspace | `.wal` structure, config, registry idea |
| 2 | Add good recipe | atomic fact/edit representation |
| 3 | Build artifact | deterministic build metadata |
| 4 | Run behavior tests | exact, negative, context, no_nan-style gates |
| 5 | Tag passing build | versioned checkpoint |
| 6 | Add bad edit | intentional regression |
| 7 | CI catches failure | failed behavior gate and changed checksum |
| 8 | Blame/bisect | identify the responsible recipe/version |
| 9 | Rollback and notes | restore last good build and generate release note |

This sequence is more convincing than listing hundreds of modules.

## Demo Script

```text
1. "WAL treats model edits as recipes, not ad-hoc patches."
2. "A build compiles recipes into a versioned artifact."
3. "Behavioral tests act as CI gates for model behavior."
4. "Now I introduce a bad edit on purpose."
5. "The gate fails, so the release does not proceed."
6. "Blame and bisect identify which recipe caused the regression."
7. "Rollback restores the last good build."
8. "Release notes explain what changed in human-readable form."
9. "This is a pre-alpha workflow prototype; heavy model validation is the next milestone."
```

## Commands For Reviewers

Fast local checks:

```bash
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python -m wal validate-results experiments --fail-on-invalid
python experiments/m624_full_test_inventory.py
python experiments/m625_safe_runtime_sweep.py
python experiments/m626_technical_report.py
python experiments/m627_polished_demo_playbook.py
```

Demo:

```bash
python wal_studio_v01/demo.py
```

## What Not To Demo First

Avoid opening with:

- total experiment count as the main argument;
- large local model paths;
- GPU OOM history;
- old generated badges;
- registry/package claims before explaining they are prototypes.

Those details are useful in the report, but they distract from the core product loop.

## Reviewer Questions And Answers

**Is this a finished model-editing system?**  
No. It is a pre-alpha framework prototype with a working demo path and many infrastructure prototypes.

**Are all experiments real model validations?**  
No. The corpus includes real checks, simulations, docs/meta modules, migrations, and safety-blocked scripts.

**What is the strongest technical claim?**  
WAL has an end-to-end recipe/build/test/debug/rollback workflow skeleton with auditable artifacts.

**What is the weakest technical area?**  
Cross-model scientific validation is still limited and should be tested on small text-only models next.

## Success Criteria

The demo is successful if a reviewer understands:

- what a WAL recipe is;
- how a build is tested;
- how a regression is detected;
- how blame/bisect/rollback fit together;
- why the project is honestly labeled pre-alpha.
