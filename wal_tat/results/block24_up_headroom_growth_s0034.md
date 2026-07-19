# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 96; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001393 | 74.51% | 0.000000 |
<!-- candidate_only: c4_train=0.994994, squad_train=1.001393, vendor_code_train=0.982034 -->
| candidate_bf16_mlp | commit | 1.001550 | 74.51% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995157, squad_train=1.001550, vendor_code_train=0.983244 -->

Fresh verify: pass; c4_train=0.995002, squad_train=1.001428, vendor_code_train=0.981993.
