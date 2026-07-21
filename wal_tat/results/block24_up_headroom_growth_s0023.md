# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 768; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002069 | 71.58% | 0.000000 |
<!-- candidate_only: c4_train=0.995116, squad_train=1.002069, vendor_code_train=0.981897 -->
| candidate_bf16_mlp | rollback | 1.002146 | 70.80% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995248, squad_train=1.002146, vendor_code_train=0.982426 -->

Fresh verify: pass; c4_train=0.995127, squad_train=1.002080, vendor_code_train=0.981922.
