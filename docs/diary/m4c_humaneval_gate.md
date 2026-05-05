# M4C HUMANEVAL GATE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m4c_humaneval_gate.

## Configuration
iters=20, threshold=0.0

## Method / What was tested
See `experiments/m4c_humaneval_gate.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m4c_humaneval_gate.py`

## Notes from dev_diary_ru.md
```
- После текстового quality gate был добавлен кодовый gate через `lm_eval` и HumanEval.
- Для запуска понадобился явный unsafe-code path: `confirm_run_unsafe_code=True`.
- Чтобы скрипт был самодостаточным, в `m4c_humaneval_gate.py` был добавлен внутренний `HF_ALLOW_CODE_EVAL=1`.

Два уже подтверждённых запуска:
```

## Known Results (from project context)

**Result:** HumanEval gate. limit=20: dense 0.7, routed 0.7. limit=164: dense 0.7317, routed 0.7317.

**Notes:** Code quality does not degrade. Dense elapsed: 212.7s, routed: 174.7s.
