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
