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
`paired_checkpoint_audit.py` on a frozen holdout. It stores per-window BF16 and
candidate NLL sums and uses a deterministic circular moving-block bootstrap.
The block resampling is required because adjacent frozen windows come from a
contiguous token stream and are not independent documents. A checkpoint is
`confidence_passed` only when every domain's upper ratio confidence bound is
within the supplied cumulative gate.
