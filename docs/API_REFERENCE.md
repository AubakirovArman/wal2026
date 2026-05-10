# WAL API Reference

Status: pre-alpha.

## Packaged CLI

WAL exposes two explicit command surfaces:

```bash
python -m wal core --help
python -m wal studio --help
```

### Core

`wal core` wraps the legacy WAL core operations:

- `encode`
- `decode`
- `grammar`
- `compress`
- `hierarchy`
- `torch`
- `debug`
- `library`
- `backend`
- `meta`
- `export`
- `merge`
- `pipeline`
- `validate-results`

### Studio

`wal studio` currently wraps the pre-alpha artifact registry subset:

- `init <base_model>`
- `edit add <recipe_file>`
- `status`
- `tag <name> [build_id]`
- `rollback <tag>`

The full WeightOps workflow is represented by the scripted demo and milestone experiments, not yet by a stable Python API.

## Python Modules

Stable-enough imports for tests and experiments:

```python
from wal.results import validate_results
from wal.cross_model_protocol import discover_candidates
from wal.encoder import wal_encode_scalar
from wal.decoder import wal_decode_scalar_torch
```

## Compatibility

Legacy top-level commands remain supported:

```bash
wal validate-results experiments --fail-on-invalid
wal encode --help
wal init local-demo-model
```

## Non-API

Historical examples that mention `WALProject`, direct LoRA training helpers, or full `build/test/diff/blame/bisect` Python APIs are product-direction notes, not stable packaged APIs.
