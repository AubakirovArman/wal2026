# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001951 | 4.30% | 0.000000 |
<!-- candidate_only: c4_train=0.995259, squad_train=1.001951, vendor_code_train=0.982539 -->
| candidate_bf16_mlp | commit | 1.002039 | 4.30% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995397, squad_train=1.002039, vendor_code_train=0.982916 -->

Fresh verify: pass; c4_train=0.995277, squad_train=1.001949, vendor_code_train=0.982523.
