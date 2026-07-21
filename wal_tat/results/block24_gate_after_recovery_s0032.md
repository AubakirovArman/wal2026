# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002118 | 5.76% | 0.000000 |
<!-- candidate_only: c4_train=0.995286, squad_train=1.002118, vendor_code_train=0.983520 -->
| candidate_bf16_mlp | rollback | 1.002191 | 5.57% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995458, squad_train=1.002191, vendor_code_train=0.984196 -->

Fresh verify: pass; c4_train=0.995340, squad_train=1.002101, vendor_code_train=0.983510.
