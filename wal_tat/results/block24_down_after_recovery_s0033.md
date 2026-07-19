# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002020 | 9.77% | 0.000000 |
<!-- candidate_only: c4_train=0.995275, squad_train=1.002020, vendor_code_train=0.983637 -->
| candidate_bf16_mlp | rollback | 1.002499 | 9.67% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995353, squad_train=1.002499, vendor_code_train=0.984495 -->

Fresh verify: pass; c4_train=0.995299, squad_train=1.002073, vendor_code_train=0.983621.
