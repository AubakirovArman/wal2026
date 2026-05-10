# AIGI 1.0 Pre-Alpha

Status: **pre-alpha research SDK layer**.

AIGI is a verified memory-accumulation system built on top of WAL ideas. The current implementation does not claim AGI and does not perform real weight editing yet. It provides the first controlled loop:

```text
experience → memory candidate → tier selection → verification → commit/reject → audit log
```

## Current MVP

- `AIGISystem` Python SDK.
- `MemoryCompiler` for tier selection.
- `WALMemory` recipe ledger for WAL-compatible memory artifacts.
- Retrieval overlay for serving committed memories.
- Refusal memory tier for safety/policy memories.
- Verification gates for empty data, confidence range, secret-like tokens, contradiction, and refusal shape.
- JSONL runtime logs.

## Non-Claims

- No autonomous AGI claim.
- No real semantic weight-edit backend attached yet.
- No production-readiness claim.
- WAL recipe commits are served through retrieval overlay until a real weight backend is connected.
