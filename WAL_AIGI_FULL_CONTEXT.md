# WAL / AIGI Full Context

Date: 2026-05-10  
Status: **pre-alpha research framework prototype**  
Repository: `https://github.com/AubakirovArman/wal2026`  
Local path: `/mnt/hf_model_weights/arman/3bit/wal`

This is the single-file operational context for the WAL project and the AIGI 1.0 pre-alpha layer. It is a compact source of truth for reviewers, future agents, and development sessions. The raw corpus remains in `experiments/`, `book/`, `docs/`, `src/`, and `logs/`.

## 1. Short Positioning

WAL Studio is a **pre-alpha WeightOps research framework** for representing model behavior edits as reproducible recipes, validating them with behavioral gates, recording artifacts, debugging regressions, and rolling back unsafe changes.

AIGI 1.0 is a separate pre-alpha SDK layer on top of WAL ideas. Its current scope is **verified memory accumulation**:

```text
experience → memory candidate → tier selection → verification → commit/reject → contract check → rollback if needed
```

Current AIGI is **not** autonomous AGI, not ready for production deployment, and not a base-weight semantic editing backend yet. M693 adds real HF inference integration, M694 adds real soft-prompt adapter training, M695 adds real logit-LoRA adapter training, and M696 injects LoRA into an actual MLP module. These are still one-fact controlled gates, not production multi-fact editing or MEMIT base-weight editing. `wal_recipe` currently means a WAL-compatible recipe artifact plus retrieval overlay.

## 2. Current Metrics

| Area | Value |
|------|-------|
| Milestone scripts | 799 |
| Python scripts in `experiments/` | 821 |
| Result JSON files | 489 |
| Book entries | 634 |
| Docs files | 235 |
| Developer diary lines | 5546 |
| Python source modules | 95 |
| Maintained pytest tests | 35 |
| Safe sweep | 289 PASS / 532 BLOCKED |
| Result schema | 489 valid / 0 invalid |
| Docs smoke | 69 / 69 commands PASS |
| Legacy manifest | 821 scripts classified |
| Current public-claim-allowed experiments | 61 |

## 3. Repository Map

```text
wal/
├── README.md                         # public entrypoint
├── WAL_AIGI_FULL_CONTEXT.md          # this single-file context
├── TECHNICAL_REPORT.md               # formal technical report
├── PROJECT_SUMMARY.md                # compact project summary
├── KNOWN_ISSUES.md                   # limitations and blocked work
├── docs/
│   ├── VALIDATION_STATUS.md          # current validation ledger
│   ├── project_metrics.json          # machine-readable metrics
│   ├── dev_diary_ru.md               # main Russian developer diary
│   ├── aigi/                         # AIGI-specific docs/logs
│   └── *_protocol.md                 # runner/security/deployment/product protocols
├── src/
│   ├── wal/                          # package CLI and WAL-facing APIs
│   ├── wal_build/                    # build system pieces
│   └── aigi/                         # pre-alpha AIGI SDK
├── experiments/                      # M1-M696+ scripts and result JSON
├── book/                             # per-experiment writeups
├── logs/aigi/                        # JSONL AIGI runtime/step logs
├── wal_studio_v01/                   # demo workflow
├── tests/                            # maintained pytest suite
├── site/                             # static GitHub Pages artifact
└── archive/generated_history/        # historical generated claims/badges
```

## 4. WAL Architecture

```text
WAL Studio / WeightOps
├── Core CLI
│   ├── wal core ...
│   └── wal studio ...
├── Recipe System
│   ├── memory/edit recipes
│   ├── signing/verification
│   ├── diff and package concepts
│   └── provenance metadata
├── Build System
│   ├── deterministic artifacts
│   ├── tags
│   ├── rollback
│   └── time-travel concepts
├── CI / Testing
│   ├── exact checks
│   ├── negative/lure/context checks
│   ├── result schema validation
│   ├── no_nan and checksum gates
│   └── public-claim gates
├── Debugging
│   ├── blame
│   ├── semantic bisect
│   ├── behavioral checksum
│   └── release-note/diff summaries
├── Memory Layer
│   ├── retrieval
│   ├── arbitration
│   ├── confidence thresholds
│   ├── refusal tier
│   └── provenance
├── Deployment Prototypes
│   ├── shadow/canary/hotfix
│   ├── emergency stop
│   ├── monitoring/alerts
│   └── backup/restore contracts
└── Registry Concepts
    ├── publish/search/install/fork
    └── package verification
```

## 5. AIGI 1.0 Architecture

AIGI is currently implemented under `src/aigi/`.

```text
AIGI SDK
├── AIGISystem
│   ├── ask
│   ├── propose_memory
│   ├── compile
│   ├── commit
│   └── rollback_last
├── Model Backend
│   ├── StaticTextModelBackend
│   ├── HuggingFaceTextBackend
│   ├── SoftPromptAdapterTrainer
│   ├── LogitLoRAAdapterTrainer
│   └── ModuleLoRAAdapterTrainer
├── MemoryCompiler
│   ├── wal_recipe tier
│   ├── retrieval tier
│   ├── refusal tier
│   ├── tool tier
│   └── reject tier
├── MemoryVerifier
│   ├── empty data gate
│   ├── confidence range gate
│   ├── secret scanner
│   ├── contradiction gate
│   └── refusal shape gate
├── BehavioralContract
│   ├── must_answer
│   ├── must_not_answer
│   └── must_refuse
├── LessonExtractor
│   └── feedback/experience → MemoryCandidate
├── VerifiedLearningLoop
│   └── experience → compile → commit → contract check → rollback
└── Logs
    ├── logs/aigi/aigi_steps.jsonl
    ├── logs/aigi/m679_runtime_events.jsonl
    ├── logs/aigi/m680_100_fact_learning_loop.jsonl
    └── logs/aigi/m681_bad_memory_rejection.jsonl
```

## 6. AIGI Gates M679-M696

| Module | Purpose | Result |
|--------|---------|--------|
| M679 | AIGI SDK skeleton | PASS: 7 positive / 4 negative checks |
| M680 | 100 fact learning loop | PASS: 100 / 100 facts |
| M681 | Bad memory rejection | PASS: 20 / 20 rejected safely |
| M682 | Memory tier routing | PASS: 9 / 9 routing checks |
| M683 | Rollback MVP | PASS: 8 / 8 rollback checks |
| M684 | Behavioral contracts | PASS: 4 / 4 contract checks |
| M685 | Experience-to-memory | PASS: 8 / 8 extraction cases |
| M686 | Verified feedback loop | PASS: 25 / 25 feedback episodes |
| M687 | Contract-gated rollback | PASS: 5 / 5 rollback checks |
| M688 | Single-file context digest | PASS: 47 / 47 checks |
| M689 | Memory change budget | PASS: 7 / 7 checks |
| M690 | Risk ledger | PASS: 8 / 8 checks |
| M691 | Contract regression suite | PASS: 6 / 6 checks over 10 protected contracts |
| M692 | Commit decision report | PASS: 7 / 7 checks |
| M693 | Real HF backend gate | PASS: 9 / 9 checks on Qwen2.5-0.5B-Instruct |
| M694 | Real soft-prompt adapter | PASS: 8 / 8 checks, loss 5.6645 → ~0.0016 |
| M695 | Real logit-LoRA adapter | PASS: 8 / 8 checks, loss 2.8775 → 0.0 |
| M696 | Real module-LoRA adapter | PASS: 9 / 9 checks, `mlp.down_proj` loss 2.5523 → ~0.0005 |

AIGI validates control flow, state-management logic, one real HF inference fallback gate, and one real gradient-trained soft-prompt adapter gate, one real low-rank logit-LoRA adapter gate, and one real MLP module-LoRA injection gate. It does not yet validate multi-fact reliability or MEMIT/base-weight edits.

## 7. Current Validation Ledger

| Gate | Status | Notes |
|------|--------|-------|
| Pytest | PASS | 35 maintained tests pass |
| Result schema | PASS | 489 / 489 result JSON valid |
| M621 truthfulness audit | PASS | 55 / 55 checks |
| M622 schema gate | PASS | 489 valid / 0 invalid |
| M623 core release gate | PASS | pytest wrapper passes |
| M624 full inventory | PASS | 821 scripts, 0 parse failures |
| M625 safe runtime sweep | PASS | 289 PASS / 532 BLOCKED |
| M630 public claim checker | PASS | 0 violations |
| M631 docs command smoke | PASS | 69 / 69 commands |
| M632/M633/M635 | PASS | controlled small-model workflows |
| M634 | BLOCKED | no local Gemma-small snapshot |
| M636-M638 | PASS | 3 unique local model paths, runtime/artifact protocol only |
| M645 | SIMULATED | hard-facts hybrid backend not real yet |
| M666 | BLOCKED | real 24h runner required |
| M667 | SIMULATED | short memory sentinel only |
| M677 | PASS | experiment manifest/classification |
| M678 | PASS | M1-M50 legacy audit batch |
| M679-M687 | PASS | AIGI verified memory/feedback loop gates |
| M688 | PASS | single-file context digest |
| M689-M692 | PASS | AIGI governance: budget, risk ledger, regression suite, decision report |
| M693 | PASS | real HF inference backend: Qwen2.5-0.5B, overlay, rollback |
| M694 | PASS | real soft-prompt adapter: frozen Qwen2.5-0.5B, trained prompt, target generation |
| M695 | PASS | real logit-LoRA adapter: frozen Qwen2.5-0.5B, rank-4 logit delta, target generation |
| M696 | PASS | real module-LoRA adapter: frozen Qwen2.5-0.5B, rank-8 MLP down_proj, target generation |

## 8. Status Semantics

```text
PASS        executed and passed
FAIL        executed and failed
BLOCKED     intentionally not executed under current policy/environment
UNSUPPORTED target configuration is not supported
SIMULATED   deterministic simulation/mock, not real-world validation
DOC_ONLY    documentation-only module
NO_DATA     no measurable data produced
```

This status discipline is important. Historical optimistic artifacts such as `A+`, `complete`, and `certified` are treated as generated history, not current release claims.

## 9. Controlled Runner Taxonomy

The 532 blocked scripts are not treated as failures. They need controlled runners:

| Runner | Purpose |
|--------|---------|
| `safe_core` | fast local non-heavy scripts |
| `safe_core_with_artifact` | safe scripts with result JSON artifacts |
| `model_small` | small local model protocol |
| `gpu_or_model_controlled` | model/GPU/HF/CUDA scripts |
| `mutation_dry_run` | git/archive/delete/restore dry-runs |
| `docs_public_claims` | public docs, reports, claim generators |
| `slow_safe` | slow but safe scripts |
| `subprocess_review` | subprocess scripts needing review |
| `blocked_review` | manually reviewed blocked scripts |

## 10. Legacy Audit

M677-M678 start the **Legacy Experiment Resurrection Program**.

M1-M50 audit summary:

```text
total scripts audited: 143
controlled model/GPU runner: 133
slow runner: 3
still-valid-needs-schema-v1: 7
current public claims: 0
```

The policy is: old experiments are not deleted, but every experiment needs classification, result schema, runner assignment, and honest public-claim eligibility.

## 11. Small-Model Status

Current controlled local small-model/runtime proof has 3 unique local model paths:

```text
M632: SmolLM2-360M family workflow PASS
M633: Qwen2.5-0.5B workflow PASS
M635: TinyLlama-1.1B workflow PASS
M636-M638: aggregate cross-model runtime/artifact gates PASS
M634: Gemma-small BLOCKED until local snapshot exists
```

These gates validate local runtime/artifact lifecycle portability. They do **not** prove semantic weight-edit training across models.

## 12. Important Commands

```bash
cd /mnt/hf_model_weights/arman/3bit/wal

# Core package tests
PYTHONPATH=src:. python -m pytest -q tests

# Result schema validation
PYTHONPATH=src:. python -m wal validate-results experiments --fail-on-invalid

# Inventory and safe sweep
PYTHONPATH=src:. python experiments/m624_full_test_inventory.py
PYTHONPATH=src:. python experiments/m625_safe_runtime_sweep.py --timeout 15

# Docs and public claims
PYTHONPATH=src:. python experiments/m631_docs_command_smoke.py
PYTHONPATH=src:. python experiments/m630_public_claim_checker.py

# AIGI latest gates
PYTHONPATH=src:. python experiments/m684_aigi_behavioral_contracts.py
PYTHONPATH=src:. python experiments/m685_aigi_experience_to_memory.py
PYTHONPATH=src:. python experiments/m686_aigi_verified_feedback_loop.py
PYTHONPATH=src:. python experiments/m687_aigi_contract_gated_rollback.py
PYTHONPATH=src:. python experiments/m688_single_file_context_digest.py
PYTHONPATH=src:. python experiments/m689_aigi_memory_change_budget.py
PYTHONPATH=src:. python experiments/m690_aigi_risk_ledger.py
PYTHONPATH=src:. python experiments/m691_aigi_contract_regression_suite.py
PYTHONPATH=src:. python experiments/m692_aigi_commit_decision_report.py

# Controlled real-model gate; downloads/loads a small HF model
PYTHONPATH=src:. python experiments/m693_aigi_real_hf_backend_gate.py
PYTHONPATH=src:. python experiments/m694_aigi_real_soft_prompt_adapter.py
PYTHONPATH=src:. python experiments/m695_aigi_real_logit_lora_adapter.py
PYTHONPATH=src:. python experiments/m696_aigi_real_module_lora_adapter.py

# Demo
python wal_studio_v01/demo.py
```

## 13. Known Limitations / Non-Claims

- Not ready for production deployment.
- Not externally certified.
- Not autonomous AGI.
- AIGI does not perform LoRA/MEMIT base-weight semantic editing yet.
- `wal_recipe` is currently a recipe artifact plus retrieval overlay; M694/M695/M696 are separate adapter-training gates.
- Deployment modules are prototypes/simulations unless explicitly validated otherwise.
- Heavy GPU/HF/model scripts are blocked from safe sweep by policy.
- Real 24h soak testing is still blocked until a controlled long-duration runner exists.
- Hard-facts hybrid backend is simulated, not a real backend execution.
- Historical generated badges/reports with `A+`, `certified`, or `complete` are audit history only.

## 14. Current Strengths

- Large experiment corpus with explicit status semantics.
- Safe sweep has zero failures under current policy.
- Result schema is clean.
- Public claims are conservative and checked.
- WAL has recipe/build/test/rollback/debugging concepts.
- AIGI SDK now has verified memory accumulation, bad-memory rejection, rollback, behavioral contracts, contract-gated feedback learning, memory budgets, risk ledger, regression suite, decision reports, real HF inference, real soft-prompt adapter training, real logit-LoRA adapter training, and real module-LoRA injection.
- Docs, book, diary, logs, Pages, and result artifacts are synchronized through release gates.

## 15. Main Weak Points

- LoRA/MEMIT base-weight semantic edit backend is not attached to AIGI.
- Full cross-model semantic edit validation is not done.
- Early experiments still need progressive resurrection in batches.
- Many deployment/security modules are deterministic contracts or simulations, not production systems.
- Retrieval/memory routing is currently deterministic SDK logic, not a learned robust router.
- External baselines like LoRA-only, RAG-only, and WAL-hybrid still need a clean benchmark pass.

## 16. Recommended Next Steps

1. **M697-M700**: run multi-fact module-LoRA and compare against RAG-only / soft-prompt / logit-LoRA baselines.
2. **M701-M704**: run RAG-only vs WAL-recipe vs LoRA baseline on a small controlled benchmark.
3. **M705-M708**: add long-running governed feedback-loop stability checks.
4. Continue legacy audit batches: M51-M100, M101-M150, then critical old failures.
5. Add a real long-duration runner for M666/M667.
6. Keep `WAL_AIGI_FULL_CONTEXT.md` updated through M688 or a successor digest gate.

## 17. Canonical Source Files

```text
README.md
PROJECT_SUMMARY.md
TECHNICAL_REPORT.md
KNOWN_ISSUES.md
docs/VALIDATION_STATUS.md
docs/project_metrics.json
docs/dev_diary_ru.md
docs/aigi/README.md
docs/aigi/dev_diary_ru.md
docs/aigi/test_log.md
experiments/experiments_manifest.json
experiments/m625_safe_runtime_sweep_results.json
experiments/m631_docs_command_smoke_results.json
site/index.html
```

## 18. One-Line Summary

WAL/AIGI is currently a **well-instrumented pre-alpha research platform** for verified model-memory workflows: strong on auditability, gates, rollback, and documentation; still pending LoRA/MEMIT base-weight edit backend, production deployment, and stronger external baselines.
