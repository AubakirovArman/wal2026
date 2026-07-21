# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 96; linked down groups: 96.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002096 | 8.79% | 0.000000 |
<!-- candidate_only: c4_train=0.995399, squad_train=1.002096, vendor_code_train=0.982780 -->
| candidate_bf16_mlp | commit | 1.001952 | 8.79% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995440, squad_train=1.001952, vendor_code_train=0.983257 -->

Fresh verify: pass; c4_train=0.995440, squad_train=1.001895, vendor_code_train=0.983275.
