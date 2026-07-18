# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 0.999824 | 1.07% | 0.000000 |
<!-- candidate_only: c4_train=0.995000, squad_train=0.999824, vendor_code_train=0.980143 -->
| candidate_bf16_mlp | commit | 0.999741 | 1.07% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994988, squad_train=0.999741, vendor_code_train=0.980183 -->

Fresh verify: pass; c4_train=0.994998, squad_train=0.999791, vendor_code_train=0.980153.
