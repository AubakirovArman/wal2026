# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 0.999722 | 0.39% | 0.000000 |
<!-- candidate_only: c4_train=0.994976, squad_train=0.999722, vendor_code_train=0.980100 -->
| candidate_bf16_mlp | commit | 0.999814 | 0.39% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994980, squad_train=0.999814, vendor_code_train=0.980323 -->

Fresh verify: pass; c4_train=0.994969, squad_train=0.999766, vendor_code_train=0.980131.
