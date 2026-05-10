# WAL Project Summary

**Status:** pre-alpha research framework prototype
**Release Grade:** not assigned
**Version:** v1.4 post-audit cleanup plus small-model gates
**Date:** 2026-05-10

## Statistics

| Metric | Value |
|--------|-------|
| Experiments | 812 Python scripts |
| Results | 480 JSON result files |
| Books | 625 entries |
| Docs | 230 docs plus developer diary |
| Maintained tests | 30 passing |
| Safe runtime sweep | 284 passing, 0 failing, 528 blocked by policy |
| Small-model controlled workflows | 3 passing unique local model paths |
| Legacy audit M1-M50 | 143 scripts classified, 0 current public claims |
| AIGI verified feedback memory loop | M679-M687 passing: 100 facts, 20 bad-memory rejections, 9 routing checks, 8 rollback checks, 25 feedback episodes |

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
- M632, M633, and M635 pass controlled local small-model runtime/artifact workflows on SmolLM2-360M, Qwen2.5-0.5B, and TinyLlama-1.1B.
- M636-M638 pass with 3 unique local model paths; M634 remains `BLOCKED` until a Gemma-small snapshot is available.
- `docs/VALIDATION_STATUS.md` provides the public validation ledger and non-claims.
- M639-M644 add dirty/ambiguous/temporal/long-answer/procedural/refusal robustness corpora and routing contracts.
- M645 records hard-facts hybrid backend work as `SIMULATED`, not a real backend claim.
- M646-M651 add CI hardening corpora, long-context payloads, scoring calibration, and checksum drift checks.
- M652-M658 add static security gates for secrets, recipe injection, registry poisoning, hotfix abuse, retrieval injection, provenance tamper, and signed package verification.
- M659-M668 add deployment reality contracts; M666 remains `BLOCKED` for a real 24h runner and M667 is `SIMULATED` as a short memory sentinel.
- M669-M676 add product polish gates for CLI UX, docs consistency, demo E2E, static Pages build, pre-alpha release dry run, and public repo hardening.
- M677-M678 add the Legacy Experiment Resurrection manifest and first M1-M50 audit batch.
- M679-M687 extend the separate AIGI SDK layer with verified memory accumulation, behavioral contracts, feedback extraction, contract-gated rollback, logs, and positive/negative tests.
- M501 is correctly marked `BLOCKED` due CUDA OOM.
- M601 is correctly marked `UNSUPPORTED` for the current Qwen-VL AutoModel path.

## Honest Assessment

**Strong:** CLI, recipes, build/test/rollback concepts, result validation, debugging prototypes, audit trail, generated book/diary, safe release gates.

**Weak:** semantic weight-edit training is not yet validated across the small-model set, real GPU training remains resource-bound, deployment modules are mostly simulations/prototypes, historical meta files still contain generated optimism in older entries.

**Status:** Research-grade pre-alpha WeightOps framework prototype with explicit known issues and release gates.
