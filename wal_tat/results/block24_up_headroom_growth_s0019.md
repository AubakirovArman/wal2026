# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001539 | 68.46% | 0.000000 |
<!-- candidate_only: c4_train=0.995267, squad_train=1.001539, vendor_code_train=0.981978 -->
| candidate_bf16_mlp | commit | 1.001742 | 68.46% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995437, squad_train=1.001742, vendor_code_train=0.982695 -->

Fresh verify: pass; c4_train=0.995283, squad_train=1.001555, vendor_code_train=0.981968.
