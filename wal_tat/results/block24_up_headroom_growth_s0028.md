# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.up_proj`; new groups: 384; linked down groups: 8192.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.002056 | 73.93% | 0.000000 |
<!-- candidate_only: c4_train=0.995258, squad_train=1.002056, vendor_code_train=0.982312 -->
| candidate_bf16_mlp | rollback | 1.002161 | 73.54% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995440, squad_train=1.002161, vendor_code_train=0.983087 -->

Fresh verify: pass; c4_train=0.995249, squad_train=1.002039, vendor_code_train=0.982267.
