# WAL Project Summary

**Status:** pre-alpha research framework prototype
**Release Grade:** not assigned
**Version:** v1.4 post-audit cleanup
**Date:** 2026-05-09

## Statistics

| Metric | Value |
|--------|-------|
| Experiments | 756 Python scripts |
| Results | 424 JSON result files |
| Books | 569 entries |
| Docs | 221 markdown docs plus developer diary |
| Maintained core tests | 12 passing |
| Safe runtime sweep | 233 passing, 0 failing, 523 blocked by policy |

## Key Results

- Core package APIs pass the maintained pytest suite.
- `wal validate-results` validates the result corpus under `wal.results.v1` compatibility rules.
- M624 compiles/inventories every experiment script and reports 0 compile failures.
- M625 executes all safe non-heavy/non-mutating experiments in order and records the sweep.
- M626 adds the canonical technical report for honest public framing.
- M627 adds the polished demo playbook for reviewer walkthroughs.
- M628 maps all blocked scripts into controlled runner categories.
- M629 defines the seven-runner hardening matrix.
- M630 verifies public claims across release-facing files.
- M631 smoke-tests fast reviewer documentation commands.
- M501 is correctly marked `BLOCKED` due CUDA OOM.
- M601 is correctly marked `UNSUPPORTED` for the current Qwen-VL AutoModel path.

## Honest Assessment

**Strong:** CLI, recipes, build/test/rollback concepts, result validation, debugging prototypes, audit trail, generated book/diary, safe release gates.

**Weak:** full cross-model WAL workflow not yet validated, real GPU training remains resource-bound, deployment modules are mostly simulations/prototypes, historical meta files still contain generated optimism in older entries.

**Status:** Research-grade pre-alpha WeightOps framework prototype with explicit known issues and release gates.
