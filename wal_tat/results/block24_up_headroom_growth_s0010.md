# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.002592 | 57.52% | 0.000000 |
<!-- candidate_only: c4_train=0.995558, squad_train=1.002592, vendor_code_train=0.981775 -->
| candidate_bf16_mlp | rollback | 1.002555 | 57.52% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995671, squad_train=1.002555, vendor_code_train=0.982546 -->
