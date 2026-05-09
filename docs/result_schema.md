# WAL Result Schema v1

WAL result files use a conservative publication schema for experiment outputs.

## Required Fields

- `schema_version`: use `wal.results.v1` for normalized results.
- `status`: one of `PASS`, `FAIL`, `BLOCKED`, `UNSUPPORTED`, `SIMULATED`, `NO_DATA`, `DOC_ONLY`.
- `pass`: legacy boolean compatibility field derived from `status`.

## Status Meaning

- `PASS`: completed and met the stated check.
- `FAIL`: completed but failed the stated check.
- `BLOCKED`: could not complete due to an external blocker such as resource limits.
- `UNSUPPORTED`: attempted path is not supported by the current dependency/model stack.
- `SIMULATED`: prototype simulation, not a real production validation.
- `NO_DATA`: no usable result was produced.
- `DOC_ONLY`: documentation/meta artifact, not behavioral validation.

## Legacy Results

Older list-shaped results are wrapped as:

```json
{
  "schema_version": "wal.results.v1",
  "status": "PASS",
  "pass": true,
  "record_count": 2,
  "records": []
}
```

The original records stay intact under `records`.

## Validation

```bash
wal validate-results experiments --output experiments/m544_result_validation_results.json
```

Use `--fail-on-invalid` in release gates.
