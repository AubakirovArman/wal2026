# M43ZK VRE LAYER0 SELECTIVE

## Date
2026 (exact date from git log or experiment run)

## Goal
M43zk: VRE only on q/k/v/gate/up in layer 0, scalar on o_proj/down_proj and all smooth.

## Configuration
K=128, iters=5, block_size=4, num_steps=0, threshold=0.0

## Method / What was tested
See `experiments/m43zk_vre_layer0_selective.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43zk_vre_layer0_selective.py`
- `experiments/m43zk_vre_layer0_selective.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.