# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002050 | 75.78% | 0.000000 |
<!-- candidate_only: c4_train=0.995383, squad_train=1.002050, vendor_code_train=0.983863 -->
| candidate_bf16_mlp | rollback | 1.002361 | 75.68% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995642, squad_train=1.002361, vendor_code_train=0.985130 -->

Fresh verify: pass; c4_train=0.995406, squad_train=1.001972, vendor_code_train=0.983911.
