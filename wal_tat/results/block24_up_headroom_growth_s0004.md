# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000666 | 41.89% | 0.000000 |
<!-- candidate_only: c4_train=0.995526, squad_train=1.000666, vendor_code_train=0.981178 -->
| candidate_bf16_mlp | commit | 1.000608 | 41.89% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995697, squad_train=1.000608, vendor_code_train=0.981494 -->

Fresh verify: pass; c4_train=0.995732, squad_train=1.000581, vendor_code_train=0.981559.
