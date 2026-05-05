# M266 — Full WAL CLI Smoke Test

**Date:** 2026-04-20
**File:** `experiments/m266_cli_smoke_test.py`

## Purpose

Build System MVP: test complete CLI workflow:
```bash
wal init → edit add → build → test → tag → diff → rollback → status
```

## Results

```
✅ wal init         → project initialized
✅ wal edit add ×4  → 4 recipes added
✅ wal build ×2     → builds #0 (14.5s) and #1 (13.6s)
✅ wal test         → CI score 0.65, verdict FAIL (negative/PPL)
✅ wal tag v1/v2    → tags created
✅ wal diff v1 v2   → shows Spain added
✅ wal rollback v1  → rolled back to 3 recipes
✅ wal status       → shows project state
```

Build hashes:
- v1: `ee0930d53755fc0d` (3 recipes)
- v2: `2e1d1b518368dbfa` (4 recipes)

## Conclusion

✅ **CLI MVP works end-to-end.** All 8 commands functional. Build System MVP is ready for demo.
