# M636 — Cross-Model Recipe Replay

Date: 2026-05-10
Status: BLOCKED  
Result: `experiments/m636_cross_model_recipe_replay_results.json`

## Purpose

Replay one recipe contract across multiple model families.

## Result

- Required real passes: `3`
- Real passes: `1`
- Blocked inputs: `3`
- Reason: `NEEDS_THREE_REAL_SMALL_MODEL_WORKFLOWS`

## Outcome

Replay remains blocked because only M633 has a real small-model workflow result. The gate requires three passing model-family workflows.
