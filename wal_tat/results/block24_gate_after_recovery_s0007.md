# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.gate_proj`; new groups: 192; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000937 | 1.56% | 0.000000 |
<!-- candidate_only: c4_train=0.994971, squad_train=1.000937, vendor_code_train=0.981138 -->
| candidate_bf16_mlp | commit | 1.000981 | 1.56% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994978, squad_train=1.000981, vendor_code_train=0.981334 -->

Fresh verify: pass; c4_train=0.994993, squad_train=1.000926, vendor_code_train=0.981089.
