# Known Issues

WAL is a pre-alpha research framework. The repository contains real code, demos, generated reports, historical experiment logs, and simulations. Historical files may contain optimistic generated labels; current public positioning is conservative.

## Current Limitations

- **Production readiness**: WAL is not ready for production use; deployment modules are prototypes/simulations unless explicitly marked otherwise.
- **Real GPU inference**: M501 is `BLOCKED` by CUDA OOM on the available Kimi-K2-Thinking setup.
- **Qwen-VL inference**: M601 is `UNSUPPORTED` with the current `AutoModelForCausalLM` path because the local Qwen3VL config is not supported by that model class.
- **Cross-model validation**: M632/M633/M635 and M636-M638 now pass a controlled runtime/artifact protocol on three small local text models, but semantic weight-edit training across those models is still not validated.
- **Historical claims**: generated milestone artifacts may mention `A+`, `complete`, or `certified`; treat these as historical meta outputs, not release claims.
- **Sweep blocking**: M624/M625 deliberately block heavy GPU/model-loading, git-mutating, destructive, and public-doc generator scripts from automated execution. Their status is `BLOCKED` by policy, not evidence of runtime failure.

## Release Gates

- `python -m pytest -q tests`
- `wal validate-results experiments --fail-on-invalid`
- `python experiments/m510_naming_convention_check.py`
- `python experiments/m518_automated_test_suite.py`
- `python experiments/m544_result_validation.py`
- `python experiments/m624_full_test_inventory.py`
- `python experiments/m625_safe_runtime_sweep.py --timeout 15`
- `python experiments/m630_public_claim_checker.py`
- `python experiments/m631_docs_command_smoke.py`

See `docs/VALIDATION_STATUS.md` for the current public validation ledger.
