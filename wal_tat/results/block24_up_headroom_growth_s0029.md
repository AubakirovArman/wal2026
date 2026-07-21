# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.002272 | 73.93% | 0.000000 |
<!-- candidate_only: c4_train=0.995304, squad_train=1.002272, vendor_code_train=0.982589 -->
| candidate_bf16_mlp | rollback | 1.002342 | 73.93% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995459, squad_train=1.002342, vendor_code_train=0.983773 -->
