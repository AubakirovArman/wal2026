# M664 — Rollback Under Load

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m664_rollback_under_load_results.json`

## Purpose

Check rollback routing during synthetic request load.

## Result

- Requests: `100`
- Rollback at request: `40`
- Failures: `0`

## Outcome

Post-rollback requests route to the restored good version.
