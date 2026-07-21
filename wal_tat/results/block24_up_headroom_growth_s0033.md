# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001293 | 74.41% | 0.000000 |
<!-- candidate_only: c4_train=0.994978, squad_train=1.001293, vendor_code_train=0.982052 -->
| candidate_bf16_mlp | commit | 1.001286 | 74.41% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995247, squad_train=1.001286, vendor_code_train=0.983326 -->

Fresh verify: pass; c4_train=0.995197, squad_train=1.001300, vendor_code_train=0.983330.
