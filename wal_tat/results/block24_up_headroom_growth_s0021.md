# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001412 | 70.02% | 0.000000 |
<!-- candidate_only: c4_train=0.995008, squad_train=1.001412, vendor_code_train=0.981573 -->
| candidate_bf16_mlp | commit | 1.001398 | 70.02% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995068, squad_train=1.001398, vendor_code_train=0.982234 -->

Fresh verify: pass; c4_train=0.995090, squad_train=1.001392, vendor_code_train=0.982193.
