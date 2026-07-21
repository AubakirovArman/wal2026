# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 192; linked down groups: 192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001979 | 8.69% | 0.000000 |
<!-- candidate_only: c4_train=0.995277, squad_train=1.001979, vendor_code_train=0.982483 -->
| candidate_bf16_mlp | rollback | 1.002247 | 8.50% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995438, squad_train=1.002247, vendor_code_train=0.983414 -->

Fresh verify: pass; c4_train=0.995245, squad_train=1.001966, vendor_code_train=0.982474.
