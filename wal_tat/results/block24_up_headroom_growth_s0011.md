# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000820 | 59.08% | 0.000000 |
<!-- candidate_only: c4_train=0.994857, squad_train=1.000820, vendor_code_train=0.980302 -->
| candidate_bf16_mlp | commit | 1.000930 | 59.08% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994965, squad_train=1.000930, vendor_code_train=0.981071 -->

Fresh verify: pass; c4_train=0.994869, squad_train=1.000814, vendor_code_train=0.980310.
