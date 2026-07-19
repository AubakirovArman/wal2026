# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001925 | 3.91% | 0.000000 |
<!-- candidate_only: c4_train=0.995158, squad_train=1.001925, vendor_code_train=0.981917 -->
| candidate_bf16_mlp | commit | 1.001935 | 3.91% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995195, squad_train=1.001935, vendor_code_train=0.982080 -->

Fresh verify: pass; c4_train=0.995175, squad_train=1.001942, vendor_code_train=0.981884.
