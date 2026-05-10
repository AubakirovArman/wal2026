# WAL User Guide

Status: pre-alpha.

## Quick Start

```bash
pip install -e .[dev]
python -m wal --help
python -m wal core --help
python -m wal studio --help
```

## Current WAL Studio Commands

The packaged Studio CLI currently exposes the artifact/registry subset:

```bash
python -m wal studio init local-demo-model
python -m wal studio edit add examples/quickstart/facts.json
python -m wal studio status
```

The commands write to `.wal/`, which is ignored by git.

## Current WAL Core Commands

```bash
python -m wal core validate-results experiments --fail-on-invalid
python -m wal core encode --help
python -m wal core decode --help
```

Legacy top-level forms remain supported:

```bash
wal validate-results experiments --fail-on-invalid
wal encode --help
wal init local-demo-model
```

## Planned Studio Commands

The following commands are part of the WeightOps product direction and are demonstrated in `wal_studio_v01/demo.py`, but are not yet full packaged implementations:

- `wal studio build`
- `wal studio test`
- `wal studio diff`
- `wal studio blame`
- `wal studio bisect`

## Demo

```bash
python wal_studio_v01/demo.py
```

The demo shows the intended `init → recipe → build → test → bad edit → CI fail → blame/bisect → rollback → release notes` story without claiming production readiness.
