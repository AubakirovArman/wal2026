# WAL v1.4 Release Notes

## Public Status
- Pre-alpha research framework prototype
- Not ready for production use
- Not externally certified

## Highlights
- 600+ historical modules
- 770 Python experiment/prototype scripts
- 438 result JSON files
- 583 book entries
- WAL Studio v0.1 demo scaffold
- Canonical `TECHNICAL_REPORT.md`
- Public `docs/demo_playbook.md`

## Cleanup Fixes
- M501 reclassified as `BLOCKED` due to CUDA OOM
- M601 reclassified as `UNSUPPORTED` for the current Qwen-VL AutoModel path
- M510 naming check now passes with legacy naming handled explicitly
- M518 automated suite now runs maintained core pytest gate: 12 passing
- M544 result validation now uses `wal.results.v1`
- M624 compiles/inventories all 770 experiment scripts: 0 compile failures
- M625 safe runtime sweep: 246 PASS, 0 FAIL, 524 BLOCKED by policy
- M626 technical report gate: PASS
- M627 polished demo playbook gate: PASS
- M628 blocked script taxonomy: 524 assigned, 0 unassigned
- M629 controlled runner matrix: 7 runners
- M630 public claim checker: 0 violations
- M631 docs command smoke: 22/22 fast commands pass; 7 embedded model-small result files remain `BLOCKED`
- M632-M638 model-small gates: BLOCKED pending local small text-only models
- M639-M644 robustness corpus/routing gates: PASS
- M645 hard facts hybrid backend: SIMULATED contract only

## Known Issues
- See `KNOWN_ISSUES.md`
