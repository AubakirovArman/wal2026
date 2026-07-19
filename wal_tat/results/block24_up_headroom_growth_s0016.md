# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001428 | 66.89% | 0.000000 |
<!-- candidate_only: c4_train=0.994996, squad_train=1.001428, vendor_code_train=0.981100 -->
| candidate_bf16_mlp | commit | 1.001530 | 66.89% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995135, squad_train=1.001530, vendor_code_train=0.981726 -->

Fresh verify: pass; c4_train=0.994996, squad_train=1.001460, vendor_code_train=0.981141.
