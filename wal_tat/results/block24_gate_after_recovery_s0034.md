# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001720 | 5.96% | 0.000000 |
<!-- candidate_only: c4_train=0.995249, squad_train=1.001720, vendor_code_train=0.983393 -->
| candidate_bf16_mlp | commit | 1.001869 | 5.96% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995306, squad_train=1.001869, vendor_code_train=0.983721 -->

Fresh verify: pass; c4_train=0.995286, squad_train=1.001760, vendor_code_train=0.983387.
