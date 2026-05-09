# Next Experiment Proposal: Chunked Weight Matmul for Block-RVQ

## Current State
- **M10c end-to-end**: 1100 tok/s, 46GB VRAM peak, ppl=3.03
- **M7b per-tensor**:
  - Q proj: 21.26ms (540x slower than BF16: 0.039ms)
  - K proj: 2.63ms (82x slower than BF16: 0.032ms)
- **Bottleneck**: Block-RVQ matmul at ~500-600x slowdown vs BF16

## Proposed Experiment
**Single change**: Switch matmul strategy from `full_weight` to `chunked_weight`
- **Why**: Large weight matrices (e.g., 4096→256) don't fit in shared memory. Chunking reduces memory pressure and improves cache reuse.
- **Expected gain**: 20-40% speedup based on typical chunked matmul optimizations
- **Risk**: Low - purely a matmul strategy change, no algorithmic modifications

## Command
```bash
CUDA_VISIBLE_DEVICES=2,3,5 python experiments/m7b_runtime_speed_bench.py \
  --mode block_rvq \
  --tensors model.layers.54.self_attn.q_proj.weight,model.layers.54.self_attn.k_proj.weight \
  --configs 1x1,1x32,1x512,1x2048,4x2048 \
  --warmup 3 --iters 10 \
  --group-rows 2048 --block-size 32 --codebook-size 256 \
  --num-stages 3 --product-splits 4 \
  --transform-kind none --calibrate-stage-scales \
  --residual-correction none \
  --matmul-strategy chunked_weight \
  --out results/m7b_chunked_weight.json
```

## Success Criteria
- **Target**: tok/s > 1500 (≥35% improvement from 1100)
- **Acceptable**: tok/s > 1300 with similar rel_mse (<0.07)
- **Measure**: Both throughput and per-layer rel_mse must be tracked