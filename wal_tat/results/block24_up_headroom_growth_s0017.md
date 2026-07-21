# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.002134 | 66.89% | 0.000000 |
<!-- candidate_only: c4_train=0.995164, squad_train=1.002134, vendor_code_train=0.981208 -->
| candidate_bf16_mlp | rollback | 1.002206 | 66.89% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995296, squad_train=1.002206, vendor_code_train=0.981928 -->
