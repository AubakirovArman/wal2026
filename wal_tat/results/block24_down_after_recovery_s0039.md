# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002033 | 10.35% | 0.000000 |
<!-- candidate_only: c4_train=0.995409, squad_train=1.002033, vendor_code_train=0.983869 -->
| candidate_bf16_mlp | rollback | 1.002409 | 10.25% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995600, squad_train=1.002409, vendor_code_train=0.984847 -->

Fresh verify: pass; c4_train=0.995373, squad_train=1.002047, vendor_code_train=0.983850.
