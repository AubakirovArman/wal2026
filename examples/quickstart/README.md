# WAL Studio Quickstart Example

This example exercises the current pre-alpha WAL Studio artifact workflow without downloading model weights.

```bash
python -m wal studio init local-demo-model
python -m wal studio edit add examples/quickstart/facts.json
python -m wal studio status
```

The commands write to `.wal/`, which is intentionally ignored by git.

For the full 12-step scripted demo, run:

```bash
python wal_studio_v01/demo.py
```
