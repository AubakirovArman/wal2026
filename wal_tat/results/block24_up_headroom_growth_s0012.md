# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 1536; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001272 | 60.64% | 0.000000 |
<!-- candidate_only: c4_train=0.995037, squad_train=1.001272, vendor_code_train=0.981142 -->
| candidate_bf16_mlp | commit | 1.001290 | 60.64% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995332, squad_train=1.001290, vendor_code_train=0.982249 -->

Fresh verify: pass; c4_train=0.995029, squad_train=1.001221, vendor_code_train=0.981257.
