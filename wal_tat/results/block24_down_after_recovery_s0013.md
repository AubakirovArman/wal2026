# WAL-TAT atomic compensation window

Candidate: `model.layers.24.mlp.down_proj`; new groups: 384; linked down groups: 384.

| Arm | Gate | Worst ratio | Candidate coverage | Down churn |
|---|---|---:|---:|---:|
| candidate_only | commit | 1.001564 | 5.76% | 0.000000 |
<!-- candidate_only: c4_train=0.995238, squad_train=1.001564, vendor_code_train=0.981940 -->
| candidate_bf16_mlp | commit | 1.001450 | 5.76% | 0.000000 |
<!-- candidate_bf16_mlp: c4_train=0.995389, squad_train=1.001450, vendor_code_train=0.982288 -->

Fresh verify: pass; c4_train=0.995395, squad_train=1.001428, vendor_code_train=0.982313.
