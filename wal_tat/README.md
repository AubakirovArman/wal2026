# WAL-TAT

WAL-TAT is an experimental library for **transactional adaptive
ternarization** of language-model matrices. It is neither ordinary one-shot PTQ
nor full-model BitNet training:

1. collect activation and loss-gradient moments on frozen calibration data;
2. rank g128 groups by estimated causal damage;
3. snapshot a small candidate set in WAL v2;
4. run a local soft-to-hard ternary QAT transaction;
5. evaluate hard codes on frozen general and code domains;
6. commit only if every NLL ratio passes the gate, otherwise restore exactly;
7. repeat, shrinking or changing the transaction when the frontier is reached.

The hard codebook is `{-1, 0, +1}`. That is logically
`log2(3) = 1.585` bits, but the current Prism-compatible layout stores two-bit
slots and one FP16 scale per 128 weights:

```text
physical bpw = 2 + 16/128 = 2.125
```

So the precise description is **ternary / logical 1.58-bit / physical
Q2-g128 2.125 bpw**. A partially converted model is a BF16 + Q2-g128 mixture.

## Current evidence

The accepted frontier on `Qwen/Qwen3-1.7B` has:

- all seven major matrices of layer 27 hard ternary;
- Q/K/V/O of layer 24 hard ternary plus 75.78125% `up_proj`,
  6.34765625% `gate_proj`, and 10.44921875% `down_proj`;
- 74,563,584 weights in `{-scale, 0, +scale}`;
- 11 full major matrices plus three partial matrices and 4.333956% of major matrix
  weights accepted;
- recurring validation-v3 NLL ratios `0.997193 / 1.001472 / 0.982272` on C4, SQuAD and
  PyTorch code;
- recurring validation-v4 ratios `0.997193 / 0.997775 / 0.983498` on C4, a different SQuAD
  slice and Transformers code;
- a prospectively declared non-overlapping D10 step passed an incremental
  paired upper-95 limit of `1.0002` (`1.000051` on v3 and `1.000107` on v4)
  while also passing the cumulative point budget;
- cumulative paired 95% bounds still fail the narrow SQuAD guide because of
  uncertainty inherited from the earlier frontier, so this is recurring
  validation rather than final sealed evidence;
- exact hard-forward ternary codes with smooth proxy-code gradients used only
  during backward recovery.

This is a successful partial conversion, **not a finished compressed 1.7B
checkpoint**. One of 28 decoder blocks is complete; the next has four complete
attention matrices and three partial MLP matrices accepted. The old `1.02` limit
is explicitly a local diagnostic gate,
not a safe per-block full-model budget. It was manually chosen in the first
WAL-TAT commit and is not a BitNet/Prism standard. See
[docs/STATUS_RU.md](docs/STATUS_RU.md), the
[full-model roadmap](docs/FULL_MODEL_ROADMAP_RU.md), and the campaign results in
`results/`.

## Package contents

- `quantization.py`: soft-to-hard and hard groupwise ternarization;
- `transforms.py`: deterministic g128 randomized-Hadamard transforms and a
  hard-code `FixedTernaryLinear` evaluation path;
- `scoring.py`: reconstruction and activation/Fisher causal ranking;
- `moments.py`: hooks for the required causal moments;
- `transaction.py`: exact candidate snapshots, commit, and rollback;
- `compensation.py`: structured linked MLP channel windows;
- `AtomicTernaryTransaction`: multi-matrix commit/rollback and committed-group reopening;
- rollback-safe continuous BF16 compensation windows that expose gradients only
  for linked uncommitted groups and never inflate ternary coverage;
- `wal.py`: WAL v2 with sequence numbers, SHA-256 hash chain, and optional fsync;
- `controller.py`: multi-domain NLL quality gate;
- `AdaptiveTransactionSizer`: rollback/tight-margin shrinking and cautious
  safe-streak growth for the next sensitivity-ranked atom;
- `experiments/adaptive_campaign.py`: durable real-training campaign loop that
  runs recovery, recurring validation, adaptive resizing and exact opt-in cleanup;
- `experiments/round_robin_campaign.py`: single-session sequential orchestrator
  that synchronizes several campaign frontiers before crash-safe cleanup;
- `experiments/fixed_transform_ablation.py`: checkpoint-neutral identity/RHT
  comparison on untouched BF16 matrices with fixed-seed holdout replication;
- `experiments/rht_proxy_recovery.py`: development-only hard-forward recovery
  of a frozen transform artifact without mutating the accepted frontier;
- `experiments/rht_artifact_audit.py`: cumulative BF16 and incremental frontier
  paired audit for a fixed transform artifact;
- `experiments/tail_initializer_ablation.py`: checkpoint-neutral sensitivity
  decile stress test with absmean, threshold-LS and activation-WLS initializers;
- `experiments/partial_initializer_proxy_recovery.py`: candidate-only
  hard-forward proxy/scale recovery with norms and prior commits frozen;
- `experiments/commit_partial_initializer_artifact.py`: hash-checked prospective
  dual-gate commit that publishes a new checkpoint without mutating its parent;
- `orchestration.py`: checkpoint hashing, common-frontier validation, atomic
  frontier synchronization, and active-worker detection;
- `evaluation.py`: deterministic HF-style before/after NLL and PPL evaluator.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest -q
python examples/toy_transaction.py
```

The package intentionally excludes checkpoints, caches, GGUF files, and raw
datasets from Git.

## Minimal use

```python
import torch
from wal_tat import (
    HashChainWAL, RatioGate, TransactionController,
    TransactionalTernaryMatrix, select_group_mask,
)

matrix = TransactionalTernaryMatrix(linear.weight, group_size=128)
scores = ...  # activation_fisher_group_damage(...)
mask = select_group_mask(scores, count=4096, strategy="lowest")

controller = TransactionController(
    HashChainWAL("runs/transactions.jsonl"),
    "model.layers.27.mlp.down_proj",
    RatioGate({"wiki": 1.02, "code": 1.02}),
)
controller.begin(matrix, mask, selector="causal_fisher")

# Optimize candidate groups while advancing the soft-to-hard schedule.
# Then evaluate strict hard weights on the same frozen suites.
decision = controller.decide(
    matrix,
    baseline={"wiki": 3.60, "code": 3.74},
    candidate={"wiki": 3.58, "code": 3.79},
)
```

WAL v2 records intent before applying commit/rollback and detects modification
or reordering. It does not contain the large weight payload and therefore does
not replace a model checkpoint. Recovery uses the verified log to locate the
last completed boundary, reloads the corresponding checkpoint, and reruns an
incomplete transaction.

The `1.02` value in the minimal example is an experimental micro-gate chosen by
this project. It is not a BitNet/Prism standard. Full-model acceptance must use
an independently defined cumulative NLL/PPL/task budget. Because
`PPL = exp(NLL)`, a 2% NLL increase is not a 2% PPL increase; at NLL 3.5 it is
approximately 7.25% PPL.
