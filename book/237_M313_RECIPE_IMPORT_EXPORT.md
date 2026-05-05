# M313 — Recipe Import/Export

## Date
2026-05-03

## Hypothesis
Recipes can be exchanged in JSON and CSV formats.

## Method
Export to JSON and CSV, import back, verify.

## Results
- JSON: 291 bytes
- CSV: 142 bytes (2× more compact)
- Both imports verified correct

## Verdict
✅ **CONFIRMED** — JSON and CSV import/export works.

## Integration
CLI supports `wal import` and `wal export` commands.
