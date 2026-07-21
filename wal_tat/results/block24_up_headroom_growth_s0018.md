# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001536 | 67.68% | 0.000000 |
<!-- candidate_only: c4_train=0.995070, squad_train=1.001536, vendor_code_train=0.981064 -->
| candidate_bf16_mlp | commit | 1.001567 | 67.68% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995195, squad_train=1.001567, vendor_code_train=0.981784 -->

Fresh verify: pass; c4_train=0.995082, squad_train=1.001520, vendor_code_train=0.981120.
