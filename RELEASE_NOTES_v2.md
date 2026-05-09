# WAL v1.4 Release Notes

## Public Status
- Pre-alpha research framework prototype
- Not ready for production use
- Not externally certified

## Highlights
- 600+ historical modules
- 756 Python experiment/prototype scripts
- 424 result JSON files
- 569 book entries
- WAL Studio v0.1 demo scaffold
- Canonical `TECHNICAL_REPORT.md`
- Public `docs/demo_playbook.md`

## Cleanup Fixes
- M501 reclassified as `BLOCKED` due to CUDA OOM
- M601 reclassified as `UNSUPPORTED` for the current Qwen-VL AutoModel path
- M510 naming check now passes with legacy naming handled explicitly
- M518 automated suite now runs maintained core pytest gate: 12 passing
- M544 result validation now uses `wal.results.v1`
- M624 compiles/inventories all 756 experiment scripts: 0 compile failures
- M625 safe runtime sweep: 233 PASS, 0 FAIL, 523 BLOCKED by policy
- M626 technical report gate: PASS
- M627 polished demo playbook gate: PASS
- M628 blocked script taxonomy: 523 assigned, 0 unassigned
- M629 controlled runner matrix: 7 runners
- M630 public claim checker: 0 violations
- M631 docs command smoke: 8/8 fast commands pass

## Known Issues
- See `KNOWN_ISSUES.md`
