# Blocked Script Taxonomy

Date: 2026-05-09
Source: `experiments/m624_full_test_inventory_results.json`

## Summary

- Total scripts: `800`
- Blocked scripts: `521`
- Assigned scripts: `521`
- Unassigned scripts: `0`

## Runner Classes

| Runner | Scripts | Purpose |
|--------|---------|---------|
| `DOCS_PUBLIC_CLAIMS` | 90 | Docs and public-claim generators behind truthfulness gates. |
| `GPU_HEAVY` | 385 | CUDA/Triton/local-model scripts with explicit hardware requirements. |
| `MODEL_CONTROLLED` | 3 | Model/tokenizer/dataset loading under pinned small/medium model protocols. |
| `MUTATION_DRY_RUN` | 37 | Git/archive/destructive operations in temp repos or temp directories only. |
| `SLOW_PROFILE` | 1 | Timeout-prone scripts measured in slow profiling suite. |
| `SUBPROCESS_REVIEW` | 5 | Scripts that spawn commands and need command-level review. |

## Reason Counts

| Reason | Count | Runner |
|--------|-------|--------|
| `archive_mutation` | 6 | `MUTATION_DRY_RUN` |
| `cuda` | 305 | `GPU_HEAVY` |
| `dataset_load` | 159 | `MODEL_CONTROLLED` |
| `destructive_file_op` | 10 | `MUTATION_DRY_RUN` |
| `destructive_shell_op` | 3 | `MUTATION_DRY_RUN` |
| `device_map` | 268 | `GPU_HEAVY` |
| `git_metadata_generator` | 1 | `MUTATION_DRY_RUN` |
| `git_mutation` | 16 | `MUTATION_DRY_RUN` |
| `hf_download` | 3 | `MODEL_CONTROLLED` |
| `local_model_path` | 243 | `GPU_HEAVY` |
| `mass_export` | 4 | `DOCS_PUBLIC_CLAIMS` |
| `mass_regeneration` | 4 | `DOCS_PUBLIC_CLAIMS` |
| `mass_rewrite` | 2 | `DOCS_PUBLIC_CLAIMS` |
| `merge_simulation_or_mutation` | 4 | `MUTATION_DRY_RUN` |
| `model_artifact` | 33 | `GPU_HEAVY` |
| `model_load` | 313 | `MODEL_CONTROLLED` |
| `public_claim_generator` | 58 | `DOCS_PUBLIC_CLAIMS` |
| `public_doc_generator` | 31 | `DOCS_PUBLIC_CLAIMS` |
| `runtime_timeout_in_safe_sweep` | 3 | `SLOW_PROFILE` |
| `self_referential_audit_script` | 2 | `INTERNAL_AUDIT` |
| `subprocess` | 15 | `SUBPROCESS_REVIEW` |
| `tokenizer_load` | 237 | `MODEL_CONTROLLED` |
| `triton` | 22 | `GPU_HEAVY` |

## Policy

- `BLOCKED` is a routing status, not a runtime failure.
- A blocked script must move into exactly one controlled primary runner before execution.
- Hardware/model runners must record model path, device, timeout, and resource failures explicitly.
- Mutation runners must use temp repos or temp directories and must not mutate the real project tree.
- Public-claim generators must pass M621/M630 truthfulness checks before artifacts are accepted.
