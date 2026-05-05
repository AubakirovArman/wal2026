# M76 WAL V1 ROUNDTRIP

## Date
2026 (exact date from git log or experiment run)

## Goal
M76: WAL v1 Round-trip Test.

## Configuration
C=16, iters=3, max_l1=8

## Method / What was tested
Tests:
1. Binary serialization round-trip (encode → serialize → deserialize → decode)
2. Text format round-trip (text → assemble → disassemble)
3. Hierarchical atom resolution consistency (fast path vs interpretable path)

## Result
PPL evaluation.
Likely negative result Has PASS/FAIL asserts

## Artifacts
- `experiments/m76_wal_v1_roundtrip.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.
## Known Results (from project context)

**Result:** 5/5 tests PASS: Binary Round-trip, Text Round-trip, Hierarchical Consistency, Binary+Hierarchical, Text→Binary→Text.

**Notes:** Binary format v1 verified. Fast path vs interpretable path match exactly.


## Extracted Metrics (from source)

- Max diff: .8
- Max diff: .8
- Max diff: .8
- Max diff: .8
- Mean diff: .8
- Mean diff: .8
