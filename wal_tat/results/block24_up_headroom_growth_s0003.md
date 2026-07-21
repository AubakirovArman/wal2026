# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 3072; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000448 | 38.77% | 0.000000 |
<!-- candidate_only: c4_train=0.995427, squad_train=1.000448, vendor_code_train=0.981118 -->
| candidate_bf16_mlp | commit | 1.000367 | 38.77% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995432, squad_train=1.000367, vendor_code_train=0.981256 -->

Fresh verify: pass; c4_train=0.995458, squad_train=1.000424, vendor_code_train=0.981213.
