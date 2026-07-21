# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 2048.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_bf16_mlp | commit | 1.001481 | 31.15% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.996022, squad_train=1.001481, vendor_code_train=0.982214 -->

Fresh verify: pass; c4_train=0.996049, squad_train=1.001499, vendor_code_train=0.982149.
