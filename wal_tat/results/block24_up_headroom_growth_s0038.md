# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001768 | 74.90% | 0.000000 |
<!-- candidate_only: c4_train=0.995212, squad_train=1.001768, vendor_code_train=0.983284 -->
| candidate_bf16_mlp | commit | 1.001939 | 74.90% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995350, squad_train=1.001939, vendor_code_train=0.984265 -->

Fresh verify: pass; c4_train=0.995233, squad_train=1.001796, vendor_code_train=0.983274.
