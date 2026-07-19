# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001584 | 65.33% | 0.000000 |
<!-- candidate_only: c4_train=0.994992, squad_train=1.001584, vendor_code_train=0.981078 -->
| candidate_bf16_mlp | commit | 1.001659 | 65.33% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995093, squad_train=1.001659, vendor_code_train=0.981423 -->

Fresh verify: pass; c4_train=0.994968, squad_train=1.001571, vendor_code_train=0.981091.
