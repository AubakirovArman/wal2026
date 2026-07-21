# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001585 | 2.93% | 0.000000 |
<!-- candidate_only: c4_train=0.995295, squad_train=1.001585, vendor_code_train=0.981976 -->
| candidate_bf16_mlp | commit | 1.001664 | 2.93% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995368, squad_train=1.001664, vendor_code_train=0.982558 -->

Fresh verify: pass; c4_train=0.995289, squad_train=1.001541, vendor_code_train=0.982049.
