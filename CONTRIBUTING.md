# Contributing to WAL

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
