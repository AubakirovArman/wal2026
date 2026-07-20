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
