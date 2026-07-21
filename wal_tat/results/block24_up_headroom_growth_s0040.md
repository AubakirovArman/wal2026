# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002035 | 75.10% | 0.000000 |
<!-- candidate_only: c4_train=0.995274, squad_train=1.002035, vendor_code_train=0.983580 -->
| candidate_bf16_mlp | commit | 1.002089 | 75.10% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995532, squad_train=1.002089, vendor_code_train=0.984696 -->

Fresh verify: pass; c4_train=0.995327, squad_train=1.002011, vendor_code_train=0.983617.
