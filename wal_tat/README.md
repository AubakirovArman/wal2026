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
Q2-g128 2.125 bpw**.  The strict branch remains BF16 + Q2-g128.  A separate
rate--distortion branch may use signed Q4-g128 (`4.125 bpw`) and, only for
groups that still miss the gate, signed Q8-g128 (`8.125 bpw`) as explicitly
counted rescue formats.

## Current evidence

There are now two independently tracked frontiers on `Qwen/Qwen3-1.7B`:

- strict checkpoint `s0053`: 76,673,024 ternary weights (`4.456565%`), one
  complete ternary decoder block, and 11 complete ternary major matrices;
- sealed-audited mixed artifact `f5f5014...`: 81,142,144 ternary, 71,191,680
  signed-Q4 and 99,324,416 signed-Q8 weights, for 251,658,240 low-bit weights
  (`14.627457%`), five complete low-bit decoder blocks and 35 complete low-bit
  major matrices.

The mixed block-24 MLP is `40.228950%` ternary and `59.771050%` Q4 at an
average `3.320421 bpw`. Including its already strict-Q2 attention, the complete
block is `55.171712%` Q2 and `44.828288%` Q4 at `3.021566 bpw`. Its new
227,968-weight Q4→Q2 step passed sealed audit-v35 at
`0.992240 / 0.944613 / 0.966782`, with incremental worst `1.004161` versus
the immutable strict source and `1.000162` versus the accepted parent.
The next layer-25 experiment kept the difficult Q4 fallback immutable and
applied a progressive `7→5→3` collapse only to the lowest-damage 1% of its
remaining Q4 groups. It converted another 683 groups (87,424 weights) to
strict Q2 and passed sealed audit-v36 at
`0.992109 / 0.943986 / 0.957007`; incremental worst was `1.003765` versus
the strict source and `1.000022` versus its accepted mixed parent.
A subsequent coverage-neutral recovery froze every code and mask and trained
only layer-24 MLP Q4 scales on code-heavy calibration. It changed 105,060
FP16 scales, selected step 320, and passed sealed audit-v37 at
`0.993252 / 0.938675 / 0.935348`; incremental worst was `1.003535` versus
the strict source and `1.000078` versus the previous mixed parent. Coverage
and projected payload are unchanged, but this artifact is the new parent for
the next reverse-compression transaction.
After its first accepted reverse-compression step,
layer 23 is `1.187388%` Q2, `61.312612%` Q4 and `37.5%` Q8 at
`5.601252 bpw`. Sealed audit-v30 ratios are
`0.992754 / 0.992227 / 0.961534`; incremental ratios versus immutable `s0053`
are `1.001822 / 1.003343 / 1.003856`, below the prospectively declared
`1.005` limit. Audit-v30 is now disclosed.

Layer 22 is a conservative deployable upper bound after its first accepted
reverse-compression step: `1.899974%` Q2, `18.099976%` Q4 and `80.000051%`
Q8 at `7.287003 bpw`. Sealed audit-v29 ratios are
`0.991622 / 0.989226 / 0.973866`, with incremental worst `1.004860`. The
high Q8 share remains explicitly scheduled for reverse Q8→Q4→Q2 distillation.

Layer 25 is now fully low-bit at `2.955882%` Q2, `17.204285%` Q4 and
`79.839834%` Q8 (`7.259476 bpw`). Its two accepted Q4→Q2 stages have
converted 1,487,744 weights into strict ternary. The latest masked progressive
stage passed sealed audit-v36 at `0.992109 / 0.943986 / 0.957007`;
incremental worst is `1.003765` versus immutable `s0053` and `1.000022`
versus the accepted parent, both below the prospectively declared `1.005`
limit.

The strict historical lineage has:

- all seven major matrices of layer 27 hard ternary;
- Q/K/V/O of layer 24 hard ternary plus 75.78125% `up_proj`,
  10.611979% `gate_proj`, and 22.949219% `down_proj`;
- 76,673,024 weights in `{-scale, 0, +scale}` at `s0053`;
- 11 full major matrices plus three partial matrices and 4.456565% of major
  matrix weights accepted;
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
  `code=0.959176`;
- audit-v11 rejected the next 262,144-weight candidate and exposed a
  pre-existing SQuAD-train generalization gap. A coverage-neutral fallback QAT
  run then changed only the still-uncommitted layer-24 MLP master weights while
  preserving every accepted ternary code, FP16 scale and mask;
- the exact frozen recovery passed one-shot audit-v12 with absolute C4/SQuAD/
  Transformers-code ratios `0.994325 / 0.995866 / 0.990334` and incremental
  upper-95 ratios `0.998983 / 0.998855 / 0.997520` versus `s0050`;
- `s0050r1` kept coverage at 74,838,016 weights, contained no BF16 residual in
  committed groups, and fresh-verifies at `wiki=0.946518` and `code=0.957625`.
  It became the quality-recovered parent for the next coverage step;
- a new disjoint audit-v13 was frozen before candidate selection. A D1
  `gate_proj + activation_wls` atom added 2,048 g128 groups with zero code
  churn. Its sealed-audit cumulative worst point ratio was `1.000515`, and its
  worst incremental upper-95 ratio was `1.000245`, below the prospectively
  declared `1.000924` limit. Atomic checkpoint `s0051` raises coverage to
  75,100,160 weights and fresh-verifies at `wiki=0.946587` and `code=0.957546`;
- an accelerated 8,192-group `down_proj` candidate passed development and all
  cumulative point gates on audit-v14, but missed the code incremental
  upper-95 limit by `0.000121`; it was rejected without changing `s0051`;
- the predeclared 4,096-group retry passed its incremental audit-v15 gate, but
  audit-v15 exposed a pre-existing `s0051` SQuAD ratio of `1.004861`. The
  candidate reached `1.005260`, above the cumulative `1.002198` guide, so it
  too was rejected;
- two coverage-neutral recoveries and fresh disjoint development/audit pairs
  restored enough cross-domain margin to retry the same frozen 4,096-group
  `down_proj` atom from `s0051r2`. It passed audit-v19 with cumulative worst
  ratio `0.995173` and incremental upper-95 worst `1.000710`, below the
  predeclared `1.001306` limit. Atomic checkpoint `s0052` therefore added
  524,288 strict-ternary weights and raised coverage to 75,624,448 weights;
- a later 8,192-group attempt passed its incremental audit-v20 gate but failed
  the cumulative SQuAD point budget inherited by that new slice, so it was
  rejected without changing coverage;
- a two-stage low-rate fallback recovery then changed only the uncommitted
  layer-24 MLP weights. Its accepted second stage kept ternary code churn at
  zero and passed audit-v22 with cumulative C4/SQuAD/NumPy-code point ratios
  `0.991022 / 1.001470 / 0.972487`; its worst incremental upper-95 ratio was
  `0.998832`, below `1.0005`;
- `s0052r1` preserves all 75,624,448 accepted ternary weights. Fresh-process
  verification gives Wiki/Code relative NLL `0.943734 / 0.953158`, and all
  committed codes remain exactly in `{-1,0,+1}`;
- the reference binary packer exactly round-tripped the real partial
  `layer24.up_proj`: 12,582,912 weights became an 8,640,072-byte file at
  `5.493210` true bpw (75.78125% Q2-g128 plus BF16 fallback and all metadata);
- proxy recovery now reports distance to the `-0.5/+0.5` code boundaries,
  code entropy/churn and scale health. These diagnostics show that proxies do
  move before churn becomes nonzero, but the first broad MLP code changes were
  harmful on the new development suite.

This is a successful partial conversion, **not a finished compressed 1.7B
checkpoint**. One decoder block is strict ternary and four of 28 are fully
low-bit. The old `1.02` limit
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
