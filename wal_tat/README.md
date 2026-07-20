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

The accepted frontier `s0050` on `Qwen/Qwen3-1.7B` has:

- all seven major matrices of layer 27 hard ternary;
- Q/K/V/O of layer 24 hard ternary plus 75.78125% `up_proj`,
  8.528646% `gate_proj`, and 10.44921875% `down_proj`;
- 74,838,016 weights in `{-scale, 0, +scale}`;
- 11 full major matrices plus three partial matrices and 4.349907% of major matrix
  weights accepted;
- recurring validation-v3 NLL ratios `0.996756 / 1.001149 / 0.981170` on C4,
  SQuAD and PyTorch code;
- recurring validation-v4 ratios `0.996756 / 0.997175 / 0.982353` on C4, a
  different SQuAD slice and Transformers code;
- a prospectively declared non-overlapping D10 step passed an incremental
  paired upper-95 limit of `1.0002` (`1.000051` on v3 and `1.000107` on v4)
  while also passing the cumulative point budget;
- cumulative paired 95% bounds still fail the narrow SQuAD guide because of
  uncertainty inherited from the earlier frontier, so this is recurring
  validation rather than final sealed evidence;
- exact hard-forward ternary codes with smooth proxy-code gradients used only
  during backward recovery.
- fresh rotating validation v5/v6 exposed a pre-existing SQuAD generalization
  gap; four broad coverage-neutral KD/scale/code/staged recovery attempts were
  rejected without publishing a checkpoint;
- a matched counterfactual audit localized most of the recoverable block-24
  damage to committed `up_proj` groups. Target-local counterfactual-teacher
  distillation produced a useful BF16 residual, then projected it into fixed
  ternary codes plus FP16 g128 scales with no persistent residual;
- the frozen strict-Q2 scale update improved all three audit-v3 point estimates.
  Its paired incremental upper-95 ratios were `1.000032 / 0.999990 / 1.000039`,
  all below the predeclared `1.0001` limit. The resulting lineage checkpoint
  `s0048r1` was fresh-loaded before its parent was deleted;
- direct target-local scale QAT with deployment-faithful FP16 scale rounding
  moved the same strict-Q2 state in the right direction, but its best frozen
  improvement was only `0.000080637`, below the predeclared `0.0001`
  publication threshold. Extending the deterministic run to 768 steps did not
  beat step 512, so no checkpoint was published;
- a behavior-gradient boundary search then tested only adjacent
  `-1 <-> 0 <-> +1` edits. A broad 512-code candidate improved every point
  estimate but failed the predeclared SQuAD upper-95 gate (`1.000153 >
  1.0001`) and was rejected. A fixed 128-group candidate passed with paired
  incremental upper-95 ratios `0.999670 / 0.999924 / 0.999033` on
  C4/SQuAD/code;
- `s0048r2` therefore changes exactly 128 committed ternary codes and 128 FP16
  group scales, keeps coverage unchanged, contains no BF16 residual, and
  fresh-verifies at `wiki=0.949896` and `code=0.961393` relative NLL;
- on the already disclosed rotating suites, v6 still passed the cumulative
  point guide (`worst=1.000922`) while v5 SQuAD improved from the earlier
  `1.005234` to `1.005052` but remained above `1.002167`. Coverage growth was
  paused while a new non-overlapping matched development path was built;
- a joint strict-Q2 MLP initializer improved its matched selection and
  confirmation slices, but was rejected because audit-v7 SQuAD exceeded the
  cumulative guide. This left the accepted checkpoint unchanged;
- target-local fallback-compensation QAT then kept every accepted group hard
  ternary while adjusting only the still-uncommitted BF16 MLP weights. The
  aggregate fallback update was `0.3151%` relative to the source weights;
- the frozen compensation state passed the independent audit-v8 on C4,
  SQuAD and PyTorch code. Absolute NLL ratios were
  `0.993307 / 0.992380 / 0.985639`; paired incremental upper-95 ratios versus
  the parent frontier were `0.998246 / 0.996783 / 0.994289`;
- `s0048r3` atomically published that state. It changed no committed mask,
  kept coverage at 74,563,584 weights, contained no BF16 residual in committed
  groups, and fresh-verifies at `wiki=0.947025` and `code=0.959163` relative
  NLL;
- after freezing a new non-overlapping audit-v9, a post-QAT scan compared the
  96 lowest-damage remaining groups of all three layer-24 MLP matrices. A
  `gate_proj` threshold-LS candidate won development, and proxy QAT safely
  retained step zero rather than publishing a worse trained snapshot;
- the exact 12,288-weight candidate passed one-shot audit-v9. Its incremental
  paired upper-95 ratios were `1.000039 / 1.000028 / 1.000000` on
  C4/SQuAD/code, below the predeclared `1.0002` limit. `s0049` therefore raises
  coverage to 74,575,872 weights and fresh-verifies at `wiki=0.947054` and
  `code=0.959188`;
- before any further candidate work, audit-v10 was frozen on three new token
  ranges and proved disjoint from all 19 retained suites. A prospective scan
  then tested 2,048 D1 groups (262,144 weights) for each remaining layer-24
  MLP matrix, 21.33x the previous atom. `gate_proj + activation_wls` won;
  proxy recovery selected step 128 with zero code churn and a worst full
  development ratio of `1.000095`;
- that exact frozen artifact passed the one-shot audit-v10. Its incremental
  paired upper-95 ratios were `1.000039 / 1.000336 / 1.000148`, below the
  predeclared size-adjusted `1.000924` limit. `s0050` therefore raises coverage
  to 74,838,016 weights and fresh-verifies at `wiki=0.947148` and
  `code=0.959176`; it is the accepted WAL-TAT frontier;
- the reference binary packer exactly round-tripped the real partial
  `layer24.up_proj`: 12,582,912 weights became an 8,640,072-byte file at
  `5.493210` true bpw (75.78125% Q2-g128 plus BF16 fallback and all metadata);
- proxy recovery now reports distance to the `-0.5/+0.5` code boundaries,
  code entropy/churn and scale health. These diagnostics show that proxies do
  move before churn becomes nonzero, but the first broad MLP code changes were
  harmful on the new development suite.

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
- `packing.py`: versioned reference binary Q2-g pack/unpack, partial BF16
  fallback, reserved-code validation and exact `true_artifact_bpw()` accounting;
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
- `experiments/build_diverse_recovery_suite.py`: offset-declared, optionally
  interleaved recovery/development suites with configurable code corpora;
- `experiments/recovery_suite_overlap_audit.py`: exact-window and declared-range
  leakage audit across recovery and validation artifacts;
- `experiments/counterfactual_bf16_restore_audit.py`: checkpoint-neutral
  localization of damage from committed and fallback regions;
- `experiments/matched_bf16_residual_recovery.py`: matched continuous control
  with raw or target-local counterfactual teachers;
- `experiments/distill_residual_to_ternary.py`: strict projection of a useful
  continuous residual into ternary codes and FP16 g128 scales;
- `experiments/counterfactual_scale_recovery.py`: deployment-faithful direct
  FP16-scale QAT against a target-local counterfactual teacher;
- `experiments/committed_mlp_initializer_ablation.py`: matched joint
  up/gate/down strict-Q2 initializer selection with disjoint confirmation;
- `experiments/mlp_fallback_compensation_recovery.py`: target-local QAT that
  freezes accepted ternary groups and trains only the uncommitted BF16 MLP
  fallback as a coverage-neutral bridge to the next conversion;
- `experiments/boundary_sparse_recode.py`: behavior-gradient ranking of sparse
  adjacent ternary-code moves with disjoint development confirmation;
- `experiments/ternary_recode_artifact_audit.py`: fresh paired cumulative and
  incremental audit of a frozen coverage-neutral Q2 artifact;
- `experiments/commit_ternary_recode_artifact.py`: atomic, hash-checked
  publication of an audited coverage-neutral child checkpoint;
- `experiments/commit_sparse_ternary_recode_artifact.py`: separate atomic
  publisher for an audited, policy-bounded sparse code-and-scale recode;
- `experiments/commit_partial_initializer_artifact.py`: hash-checked prospective
  dual-gate commit that publishes a new checkpoint without mutating its parent;
- `experiments/commit_fallback_compensation_artifact.py`: atomic publisher that
  verifies the audit lineage and permits BF16 master changes only below false
  committed masks;
- `experiments/reference_pack_checkpoint_matrix.py`: write and independently
  reload the real versioned partial/full Q2-g binary representation;
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
