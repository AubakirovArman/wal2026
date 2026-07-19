# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.000926 | 3.42% | 0.000000 |
<!-- candidate_only: c4_train=0.994969, squad_train=1.000926, vendor_code_train=0.981121 -->
| candidate_bf16_mlp | commit | 1.000913 | 3.42% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.994992, squad_train=1.000913, vendor_code_train=0.981106 -->

Fresh verify: pass; c4_train=0.994986, squad_train=1.000950, vendor_code_train=0.981069.
