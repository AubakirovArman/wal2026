# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001398 | 5.37% | 0.000000 |
<!-- candidate_only: c4_train=0.995028, squad_train=1.001398, vendor_code_train=0.981132 -->
| candidate_bf16_mlp | commit | 1.001680 | 5.37% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995155, squad_train=1.001680, vendor_code_train=0.981775 -->

Fresh verify: pass; c4_train=0.995037, squad_train=1.001403, vendor_code_train=0.981176.
