# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | rollback | 1.002279 | 73.14% | 0.000000 |
<!-- candidate_only: c4_train=0.995225, squad_train=1.002279, vendor_code_train=0.982310 -->
| candidate_bf16_mlp | rollback | 1.002317 | 73.14% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995292, squad_train=1.002317, vendor_code_train=0.982872 -->
