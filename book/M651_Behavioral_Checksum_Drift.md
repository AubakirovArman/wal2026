# M651 — Behavioral Checksum Drift

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m651_behavioral_checksum_drift_results.json`

## Purpose

Check that unchanged behavior has a stable checksum and changed behavior produces checksum drift.

## Result

- Same behavior checksum: stable
- Changed behavior checksum: drift detected
- Fixture: `corpora/behavioral_checksum_fixtures.json`

## Outcome

Behavioral checksum semantics are now covered by a deterministic fixture gate.
