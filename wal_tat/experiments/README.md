# Experiment contract

Every model-scale WAL-TAT experiment should keep the following immutable inputs
under version control:

- model and tokenizer revision;
- calibration sample IDs and tokenized-suite digest;
- RNG seed and transaction schedule;
- candidate selector and group size;
- BF16 baseline NLL/PPL for every domain;
- hard-candidate NLL/PPL and gate decision;
- codebook validation and committed coverage;
- packed bpw projection separately from training memory.

Large checkpoints, datasets, caches, generated WAL files, and GGUF artifacts are
deliberately ignored. Store their SHA-256 and reproduction command in a small
result JSON instead of committing the binaries.

For a statistically paired audit of an accepted frontier, run
`paired_checkpoint_audit.py` on a frozen suite and declare its role with
`--suite-role`. Repeatedly exposed v3/v4 suites are recurring validation, not
sealed holdout evidence. A sealed block suite must be non-overlapping, must not
select between arms, and becomes disclosed after its first inspection.

The audit stores per-window BF16 and candidate NLL sums and uses a deterministic
circular moving-block bootstrap. Block resampling is required because adjacent
frozen windows come from a contiguous token stream and are not independent
documents. A checkpoint is `confidence_passed` only when every domain's upper
ratio confidence bound is within the supplied cumulative gate.

For a micro-step whose parent frontier already has a wide cumulative interval,
declare the candidate policy before running it and keep two tests separate:

- cumulative point ratio versus raw BF16 must remain inside the coverage budget;
- the paired incremental upper-confidence ratio versus the frozen parent must
  remain inside a small predeclared margin on every recurring-validation domain.

`commit_partial_initializer_artifact.py` enforces both conditions by hash and
publishes a child checkpoint only after every required audit agrees with the
policy. This prospective incremental gate does not turn recurring validation
into sealed evidence and must never be defined after seeing the candidate.

Coverage-neutral recovery uses the same discipline. First,
`counterfactual_bf16_restore_audit.py` identifies which already committed
region actually causes a domain gap. `matched_bf16_residual_recovery.py` may
then learn a continuous control from a target-local counterfactual teacher, but
that residual is training-only. `distill_residual_to_ternary.py` must project a
passing direction back into strict ternary codes and shared FP16 scales.
`ternary_recode_artifact_audit.py` audits that frozen state in a fresh process;
only `commit_ternary_recode_artifact.py` may publish it as a lineage-linked
checkpoint. The parent checkpoint remains on disk until a separate fresh-load
verification succeeds.

`reference_pack_checkpoint_matrix.py` writes the deployment representation
without pickle/container inflation: two-bit code slots for committed groups,
one FP16 scale per group, a bit-packed partial-coverage mask and exact BF16
fallback groups. Its reported `true_bpw_file` includes the versioned binary
header and every payload byte.

`boundary_sparse_recode.py` is a coverage-neutral discrete recovery path. It
uses a target-local counterfactual teacher to accumulate a behavior gradient,
ranks only adjacent ternary-code moves, changes at most one value in each
selected g128 group and evaluates all search candidates on development data.
Only the frozen winner is evaluated on a disjoint confirmation slice. A
passing artifact still needs `ternary_recode_artifact_audit.py` and may be
published only by `commit_sparse_ternary_recode_artifact.py`, which enforces
the predeclared exact code/scale-change counts and refuses changes outside the
committed mask.

`build_matched_validation_recovery_suite.py` builds development data from the
same source families as rotating audit v5 (C4 validation, SQuAD validation and
PyTorch source) while using separately declared token ranges. It records full
source hashes, full token-stream hashes and calibration/gate intervals. This
reduces the train/validation split mismatch without exposing the actual v5/v6
windows to gradients.

`counterfactual_scale_recovery.py` can reserve disjoint development gate
slices with `--selection-gate-start/--selection-gate-sequences` and
`--confirmation-gate-start/--confirmation-gate-sequences`. Candidate steps are
selected only on the first slice; an artifact is emitted only when the frozen
step also satisfies the predeclared improvement floor on the second slice.

`counterfactual_proxy_recovery.py` is the corresponding discrete QAT path. It
restores only the target matrix for a counterfactual teacher, executes strict
hard ternary codes and FP16-rounded scales in every student forward, and uses a
temperature-controlled soft staircase only for gradients. It writes a frozen
recode artifact after disjoint selection/confirmation acceptance and never
publishes a checkpoint directly.

`committed_initializer_ablation.py` tests whether a damaged committed region
has drifted away from the geometry of the original BF16 matrix. It rebuilds
the same mask with absmean, per-group threshold/LS and activation-weighted WLS
initializers, selects on one development slice and confirms on another. The
physical representation remains strict Q2-g128 in every arm.

`committed_mlp_initializer_ablation.py` extends that control to the three MLP
matrices jointly. It can start from a frozen single-matrix artifact, enumerates
predeclared initializer combinations on selection data, and evaluates only the
winner on the disjoint confirmation slice. Passing development remains only an
artifact candidate; it cannot bypass the independent artifact audit.

`mlp_fallback_compensation_recovery.py` is a coverage-neutral QAT bridge. Its
target-local counterfactual teacher restores only the committed MLP groups,
the student keeps those groups strict ternary, and gradients update only BF16
master values whose committed masks are false. The emitted artifact records
per-matrix fallback deltas and cannot itself publish a checkpoint.

`commit_fallback_compensation_artifact.py` publishes such an artifact only
after verifying source/artifact/audit/policy hashes, exact masks, strict codes,
and the absence of committed-master changes. The child is written atomically
and must still pass a fresh-process verification before its parent is retired.
