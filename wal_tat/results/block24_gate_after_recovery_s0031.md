# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002060 | 5.57% | 0.000000 |
<!-- candidate_only: c4_train=0.995242, squad_train=1.002060, vendor_code_train=0.983635 -->
| candidate_bf16_mlp | commit | 1.002092 | 5.57% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995299, squad_train=1.002092, vendor_code_train=0.984252 -->

Fresh verify: pass; c4_train=0.995284, squad_train=1.001996, vendor_code_train=0.983644.
