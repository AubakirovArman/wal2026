"""
M399 — Contributing Guide

Generate CONTRIBUTING.md for future collaborators.
"""
content = """# Contributing to WAL

## Development Setup

```bash
git clone https://github.com/wal-project/wal
cd wal
pip install -r requirements.txt
```

## Running Experiments

```bash
python experiments/mXXX_name.py
```

Results saved to `experiments/mXXX_name_results.json`.

## Adding Book Entries

1. Create `book/MXXX_title.md`
2. Update `docs/dev_diary_ru.md`
3. Update `ROADMAP.md`

## Code Style

- Use triple-quoted docstrings
- Save results as JSON
- Print final status line

## Testing

```bash
python experiments/m391_final_health_check.py
```

## Status

Pre-alpha. All contributions welcome but expect breaking changes.
"""

with open("CONTRIBUTING.md", "w") as f:
    f.write(content)

print("✅ M399: CONTRIBUTING.md generated")
