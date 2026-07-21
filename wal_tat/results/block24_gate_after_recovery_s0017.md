# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001756 | 3.52% | 0.000000 |
<!-- candidate_only: c4_train=0.995042, squad_train=1.001756, vendor_code_train=0.981725 -->
| candidate_bf16_mlp | commit | 1.001767 | 3.52% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995222, squad_train=1.001767, vendor_code_train=0.982074 -->

Fresh verify: pass; c4_train=0.995058, squad_train=1.001762, vendor_code_train=0.981785.
