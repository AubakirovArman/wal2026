# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001996 | 10.25% | 0.000000 |
<!-- candidate_only: c4_train=0.995417, squad_train=1.001996, vendor_code_train=0.983920 -->
| candidate_bf16_mlp | rollback | 1.002342 | 10.16% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995596, squad_train=1.002342, vendor_code_train=0.984588 -->

Fresh verify: pass; c4_train=0.995392, squad_train=1.002012, vendor_code_train=0.983924.
