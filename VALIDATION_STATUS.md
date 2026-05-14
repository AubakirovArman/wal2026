# WAL Validation Status

**Project**: WAL Studio (weightops-wal)  
**Version**: 0.1.0-pre-alpha  
**Last updated**: 2026-05-10  
**Audit program**: WAL Legacy Audit v1

---

## Executive Summary

WAL is a **pre-alpha, fully instrumented research prototype** for reproducible model edit workflows. It is not production-ready. This document tracks what has been validated, what is blocked, and what remains unverified.

---

## Validated ✅

| Component | How validated | Status |
|-----------|--------------|--------|
| Core pytest suite | `pytest -q tests/` on Python 3.10–3.12 | PASS |
| Result schema | `wal validate-results experiments --fail-on-invalid` | PASS |
| Safe runtime sweep | M625: 289 safe scripts, 279 PASS, 0 FAIL | PASS |
| Full test inventory | M624: 822 scripts parsed, 0 parse failures | PASS |
| Public claim checker | M630: README claims vs result files | PASS |
| Docs smoke test | M631: all `wal` subcommands execute | PASS |
| Release truthfulness | M621: no `status=PASS` with `error=` | PASS |
| Result schema gate | M622: all JSON conform to schema | PASS |
| Core release gate | M623: release gates sequence | PASS |
| Demo playbook | `wal_studio_v01/demo.py` 12-step workflow | PASS |
| Determinism (FP32 + seed) | M243/M253/M254: bit-exact roundtrip | PASS |
| Merge audit | M244: overlay vs merge diff < threshold | PASS |
| Batch editing (layer 16) | M244: 20 facts, FP32, rehearsal | PASS |

---

## Blocked by Infrastructure 🔒

| Component | Why blocked | Resolution path |
|-----------|------------|-----------------|
| Heavy GPU inference | 8× H200 fragmented by VLLM; only GPU-5 has ~113 GB free | Target GPU-5 explicitly or CPU offload |
| Qwen-VL-32B full load | `AutoModelForCausalLM` rejects `Qwen3VLConfig` | Use `Qwen3VLForConditionalGeneration` or pure-text model |
| Kimi-K2-Thinking full load | 594 GB; fragmented GPUs prevent load | Multi-node or dedicated allocation |
| MiniMax-M2 full load | 230 GB; same fragmentation issue | Same as above |
| Cross-model full workflow | Requires all of the above + stable multi-GPU | Deferred to v0.2 |
| Production deployment | No production environment exists | Simulation only (M671–M675) |

---

## Simulated / Doc-Only 📝

| Component | What it is | Why not "validated" |
|-----------|-----------|---------------------|
| Deployment modules (M671–M675) | Dry-run simulations of packaging, registry, release | No real PyPI / Docker Hub push |
| GitHub Pages build (M674) | Static HTML generation from README | No real deployment to `gh-pages` branch |
| Badge / certificate generators | Metadata scripts for README formatting | No external validation |
| Milestone / release declarations | Project management artifacts | Subjective, not empirical |

---

## Needs Re-Audit 🔍

| Batch | Range | Count | Priority |
|-------|-------|-------|----------|
| Batch 1 | M1–M50 | ~50 | High: early weight/route/encode probes |
| Batch 2 | M51–M100 | ~50 | High: LoRA, factual edit, WAL v1 |
| Batch 3 | M101–M170 | ~70 | High: merge/reencode, retrieval, transform |
| Batch 4 | M171–M250 | ~80 | Medium: lifecycle, CI, recipe replay |
| Batch 5 | M251–M385 | ~135 | Medium: determinism, batch, DAG, wild |
| Batch 6 | M386–M675 | ~290 | Low: platform, docs, meta (many already modern) |

See [`experiments_manifest.json`](experiments_manifest.json) for per-experiment status.

---

## Known Issues (Confirmed)

1. **FP16 LoRA overflow** — M241 proved FP16 adapters corrupt edits. Fix: FP32 adapters + grad clip 1.0.
2. **Missing seed nondeterminism** — M243/M253/M254 proved seed discipline gives bit-exact results. Fix: `wal.seeds.set_seed()` everywhere.
3. **Double-LoRA forward restore bug** — M244 proved old merge failures were harness bugs, not fundamental. Fix: `merge_audit` + `no_lora_remnants`.
4. **Wrong retrieval prompt format** — pre-M242 retrieval failed due to missing `[CONTEXT]/[QUESTION]/[ANSWER]` markers. Fix: `context_injection_v2`.
5. **Activation magnitude ≠ causal layer** — old activation-guided layer selection is unreliable. Fix: causal tracing or fixed layer 16 default.
6. **Exact-only survival is weak** — M265 showed negative gate often more informative than exact/PPL. Fix: negative/context/lure CI gates.
7. **Status semantics bug** — some old scripts set `status=PASS` while `error=` non-empty. Fix: M621 truthfulness audit.

---

## Modern Rules for Re-Audit

Any experiment claiming behavioral or edit results must now pass:

```text
✓ FP32 adapters for training/edit
✓ Fixed seed (bit-exact determinism)
✓ Layer 16 default for single-layer edits
✓ Negative-aware training + context + lure
✓ Rehearsal for batch edits
✓ Merge audit (overlay vs merge diff)
✓ No LoRA remnants after restore
✓ Behavioral checksum (not just exact survival)
✓ Proper context injection format
✓ Truthful status (no PASS with error)
```

---

## Release Gates (Current)

```yaml
core-light:
  - pytest -q tests
  - python -m wal validate-results experiments --fail-on-invalid
  - python experiments/m621_release_truthfulness_audit.py
  - python experiments/m622_result_schema_gate.py
  - python experiments/m623_core_release_gate.py
  - python experiments/m624_full_test_inventory.py
  - python experiments/m626_technical_report.py
  - python experiments/m630_public_claim_checker.py
  - python experiments/m631_docs_command_smoke.py

integration-heavy:
  - python experiments/m625_safe_runtime_sweep.py --timeout 15
  - python experiments/m671_readme_claim_checker.py
  - python experiments/m672_docs_to_code_consistency.py
  - python experiments/m673_demo_script_e2e.py
  - python experiments/m674_github_pages_build.py
  - python experiments/m675_public_release_dry_run.py
```

---

## Roadmap

| Milestone | Target | Scope |
|-----------|--------|-------|
| v0.1.0-alpha | After audit completion | Clean repo, honest claims, working demo |
| v0.2.0 | Cross-model validation | Multi-model recipe portability |
| v0.3.0 | Memory router | Production-grade memory tier + retrieval |
| v1.0.0 | Production readiness | Full WeightOps compiler + deployment |

---

*This document is living. Updates happen after each audit batch.*
