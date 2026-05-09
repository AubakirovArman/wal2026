"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M473 — CONTRIBUTING.md Update

Updates contributing guide with latest workflow.
"""
import json

content = """# Contributing to WAL

## Quick Start

1. Fork the repo
2. Create a branch: `git checkout -b feature/MXXX-name`
3. Add your experiment: `experiments/mXXX_name.py`
4. Run tests: `python experiments/mXXX_name.py`
5. Update `docs/dev_diary_ru.md`
6. Submit PR

## Experiment Template

```python
import json
print("MXXX — Title")
# Your code here
with open("experiments/mXXX_name_results.json", "w") as f:
    json.dump({"pass": True}, f)
print("✅ MXXX: Done")
```

## Standards

- All experiments must produce a `_results.json` file
- Use `assert` for validation
- Print final status line with ✅
"""

with open("CONTRIBUTING.md", "w") as f:
    f.write(content)

print("=" * 60)
print("M473 — CONTRIBUTING.md UPDATE")
print("=" * 60)
print("Updated with latest workflow")

with open("experiments/m473_contributing_results.json", "w") as f:
    json.dump({"updated": True, "pass": True}, f, indent=2)

print("\n✅ M473: CONTRIBUTING.md updated")
