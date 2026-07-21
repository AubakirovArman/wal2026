# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000163 | 1.17% | 0.000000 |
<!-- candidate_only: c4_train=0.995115, squad_train=1.000163, vendor_code_train=0.980654 -->
| candidate_bf16_mlp | commit | 1.000315 | 1.17% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995615, squad_train=1.000315, vendor_code_train=0.981665 -->

Fresh verify: pass; c4_train=0.995123, squad_train=1.000165, vendor_code_train=0.980761.
