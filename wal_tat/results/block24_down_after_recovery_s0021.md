# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.002191 | 8.50% | 0.000000 |
<!-- candidate_only: c4_train=0.995251, squad_train=1.002191, vendor_code_train=0.982373 -->
| candidate_bf16_mlp | rollback | 1.002353 | 8.50% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995419, squad_train=1.002353, vendor_code_train=0.983328 -->
