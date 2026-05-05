# Concept — Dynamic Route Language (DRL)

> Modular note: new stable design fragments should go under `docs/overview/`; keep this file as the top-level concept entry point.

## Notation

- `W ∈ R^{N×K}` — a single linear weight matrix.
- Family `f` — group of matrices sharing a ladder (e.g. `attn_qk`, `attn_vo`, `mlp_up`, `mlp_down`, `lm_head`).
- Ladder `S_f = (s_f^{(1)}, …, s_f^{(L_max)}) ∈ R^{L_max}`, strictly decreasing positive scales. Shared per family (optionally per family × depth bucket).
- Max depth `L_max = 12`.
- Alphabet `A = {-1, 0, +1}`.

## Dynamic-depth route encoding

For a single scalar weight `w`:

```
r_0 = w
for i = 1..L_max:
    d_i  = sign(r_{i-1}) * 1[ |r_{i-1}| ≥ 0.5 * s_f^{(i)} ]
    r_i  = r_{i-1} - d_i * s_f^{(i)}
    if |r_i| < eps_f or i == L_max:
        L = i
        break
route(w) = (d_1, …, d_L)
```

Reconstruction:

$$ \hat w \;=\; \sum_{i=1}^{L} d_i \cdot s_f^{(i)} $$

Dynamic length means easy weights stop at `L=2` or `L=3`, while rare outliers may use up to `L=12`.

## Packing a row

Per row `W[n, :]` we store:

- `digits[n, k]` — per-position digit, each in `{-1, 0, +1}`, packed as base-3 into `uint16` (one `uint16` holds up to 10 digits; for `L_max=12` we use 2 `uint16` per weight, OR a single `uint32`).
- `stop_depth[n, k] ∈ {1, …, 12}` — packed 4 bits per weight (one `uint8` holds 2 stop-depths).
- Optional `row_scale_fp16[n]` — guarded, same as in v1 Route B, only kept when it reduces relMSE.

## ID codebook (second pass)

After step 1 every weight has a concrete finite route. Collect the multiset of routes across the whole model; take the unique set `R = {ρ_1, ρ_2, …, ρ_M}`. Empirically `M ≪ 3^{L_max}`.

Assign each unique route an integer ID `id(ρ_j) = j ∈ {0, …, M-1}`. The codebook is either:

- **Global** — one ID space for the whole model, or
- **Per-family** — separate ID spaces per family `f`, with smaller `M_f` and lower ID bit-width.

Final stored form of `W` is an integer tensor `ID[N, K]` of IDs into the codebook, plus the codebook itself (`codebook[j]` is the packed digits + stop_depth of route `ρ_j`).

## Decoding at runtime

On every matmul with `W`:

1. Look up `route_packed = codebook[ID[n, k]]` (a single gather per weight, or per row with a small LUT when IDs repeat heavily within a row).
2. Reconstruct `ŵ[n,k] = Σ d_i · s_f^{(i)}` using `L_max` fused-multiply-adds, masked by `stop_depth`.
3. Multiply by optional `row_scale[n]` and `global_scale`.

Two runtime paths, mirroring v1:

- **Reference path** (`PackedIDRouteLinear`): materialise `ŵ` to fp16, then `matmul`.
- **Fused Triton path** (`FusedIDRouteLinear`): fuse gather + reconstruction + matmul; codebook lives in shared memory.

## Why this should beat v1 on quality

- **Dynamic depth.** Outlier rows that lose quality at fixed depth 5 can extend to depth 8–12 locally, without paying for extra digits globally.
- **Per-family ladders up to L=12.** Longer ladders = finer residual control where needed.
- **Row-scale stays.** Guarded row-scale from v1 carries over as a pure win.
- **ID codebook** is an independent lever: after step 1, codebook entropy gives a diagnostic that tells us whether to merge rare routes, split hot routes, or re-fit the ladder.

## Why this can still be fast

- Fused Triton kernel is the same topology as v1, only with a gather step and up to 12 FMAs per weight instead of 5. At 70B, the matmul is memory-bound; a larger codebook gather is fine if codebook fits in L2.
- At inference we never touch the original fp16 weights — the model on disk and in VRAM is only IDs + codebook + ladders + row scales.

## Current validation snapshot (2026-04-17)

- Full raw WikiText-2 gate: dense `3.4304` vs routed `3.4350`, gap `0.0046`.
- Full HumanEval gate (`164` tasks): dense `0.7317` vs routed `0.7195`, gap `-0.0122`.
- HumanEval elapsed: `1337.0 s` dense vs `1169.0 s` routed.
- HumanEval eval peak allocated VRAM: about `44.6 / 45.9 / 44.6 GB` across the three visible GPUs.
- HumanEval route-surgery peak allocated VRAM: about `70.2 / 71.5 / 70.2 GB`.

## Why the current runtime is slower than dense

The main slowdown is not the route alphabet itself; it is the current execution path.

- In the reference runtime, every forward first reconstructs the full dense weight matrix `W` from `ids` via `codebook_sum[ids]`, then multiplies by `row_scale`, and only after that calls `F.linear`.
- That adds an extra full pass over the whole weight tensor before the actual matmul. Dense `F.linear` only reads `W`; the route runtime reads `ids`, gathers route values, writes a temporary fp16 `W`, and then reads that temporary `W` again for the GEMM.
- For large layers this is a huge bandwidth tax. Example: `mlp.up_proj` has shape `28672 x 8192`, so one fp16 materialized weight is about `470 MB`. Reconstructing it every forward means roughly: read `ids` + write temporary `W` + read temporary `W` for GEMM, i.e. well over a gigabyte of traffic before counting activations.
- This is why the slowdown is worst at small batch / short context. With `B=1, T=512`, the activation tensor is tiny compared with the cost of touching the whole weight matrix. As batch grows, that fixed per-forward reconstruction cost is amortized, so the route runtime moves from about `0.13x` dense toward about `0.83x` dense.
- The first fused Triton kernel removes explicit `W` materialization, but it still does per-element `id -> codebook` lookups inside the matmul. That access pattern is less friendly than a standard contiguous dense GEMM on tensor cores, so it still loses to the packed reference on representative layers.
- The flat route-frequency distribution also matters: because top routes are not dominant enough, a simple "cache only hot routes" shortcut cannot remove most of the memory traffic.

## Ranked next runtime hypotheses

### 1. Global sorting of route IDs

This is the cheapest idea, but likely the weakest.

- Renumbering IDs by global frequency or by route similarity is harmless and may marginally improve lookup locality.
- But the codebook itself is already tiny (`~1500` routes, one fp16 scalar per route in the current runtime), so codebook-cache locality is not the real bottleneck.
- Therefore plain global ID sorting is unlikely to produce a large speedup by itself.

### 2. Grouping routes locally by tile, not globally by model

This is the most practical next systems direction.

- The real bandwidth problem is the full `N x K` ID matrix and the per-element decode path.
- A better storage/runtime design would be tile-local: for each weight tile, store a small local route palette plus compact local indices.
- Then the kernel loads the small palette once into shared memory and applies it across the tile, instead of performing an arbitrary global `id -> codebook` lookup for every element.
- The key discriminating check for this idea is simple: measure how many unique routes appear per tile (for example `64 x 128` or `128 x 128`) on representative layers.

### 3. Runtime-aware route distillation (our analogue of bitdistill)

This is the highest-upside research direction, but heavier than the previous two.

- Instead of only distilling for quality, train a route student under both accuracy and runtime structure constraints.
- The teacher can be either the dense model or the current high-quality routed model.
- The student objective would keep layer-output quality while also encouraging runtime-friendly structure:
    - low unique-route count per tile;
    - low route entropy per tile;
    - stable route reuse across neighboring positions;
    - small local palettes that map well to shared-memory kernels.
- This would be a better analogue of bitdistill for DRL than simple post-hoc ID renumbering, because it changes the weight language itself to match hardware.

### First implemented pilot

- `src/route_distill.py` now contains a minimal tile-local route distillation helper.
- `experiments/m6_route_distill_pilot.py` runs that helper on a real encoded layer tile.
- Current pilot protocol:
    - choose a representative tile from a real layer by sampling tiles and picking one with high route diversity;
    - keep the teacher as the row-normalized dense tile;
    - keep the student constrained to a small local palette projected back to valid route values from the layer codebook;
    - optimize a small output-matching objective on sampled activations plus assignment sharpening.
- First run on `model.layers.0.self_attn.q_proj.weight` with a sampled `128 x 128` tile found `413` unique routes in the original tile.
- Distilling that tile down to `16` projected local values produced output MSE `0.0109` and weight MSE `1.11e-4` against the teacher tile.
- This is not yet good enough for direct deployment, but it does show that a tile-local student can collapse route diversity very aggressively while staying in the same rough weight regime.
- The current pilot is intentionally minimal. The next real question is whether better objectives or better initialization can push the same idea to much lower output error.

### First multi-tile sweep after palette refinement

- `experiments/m6b_route_distill_sweep.py` now sweeps several sampled high-diversity tiles and several palette sizes.
- A constrained post-projection 1D palette refinement step was added after the first pilot; this materially improved the hard student.
- First sweep on the top-3 sampled `128 x 128` tiles of `model.layers.0.self_attn.q_proj.weight` gave:
    - palette `16`: mean output MSE `9.31e-4`, mean weight MSE `6.88e-6`, mean distilled unique `15.7`;
    - palette `32`: mean output MSE `6.46e-4`, mean weight MSE `5.23e-6`, mean distilled unique `29.3`;
    - palette `64`: mean output MSE `6.04e-4`, mean weight MSE `4.83e-6`, mean distilled unique `55.0`, but a slightly worse median output MSE than palette `32`.
- Practical reading: the local RouteDistill idea is now clearly stronger than the original one-tile pilot suggested. The best current trade-off appears near palette `32`, while palette `64` shows that the objective is improved but still not perfectly stable.

### What `palette size 32` actually means

- The student does not get 32 routes globally for the whole matrix. It gets at most `32` scalar values locally inside one tile, for example one `128 x 128` block.
- Each weight in that tile is then represented by a small local index into this per-tile palette, instead of a larger global route ID into the matrix-wide codebook.
- For the current layer-0 experiments the global route vocabulary is about `1473-1593` values, which means about `11` bits per weight for the global IDs.
- A local palette with `32` entries needs only `5` bits per local index, plus a tiny palette-overhead term. On a `128 x 128` tile this lands near `5.03 bpw`, versus `11 bpw` for the global-ID path.
- So `palette size 32` is the first regime where the tile still has enough local freedom to keep output error low on the easier families, while already cutting local index traffic by a bit more than `2x`.

### First layer-family transfer check and speed proxy

- `experiments/m6c_route_distill_layer_suite.py` extends the sweep to all 7 layer-0 linear families on their top sampled high-diversity `128 x 128` tile.
- Palette `32` results by family:
    - `q_proj`: base unique `383`, output MSE `7.29e-4`, local bpw `5.03`, decode speedup `1.05x`, decode-plus-linear speedup `1.03x`;
    - `k_proj`: base unique `481`, output MSE `7.91e-3`, local bpw `5.03`, decode speedup `1.04x`, decode-plus-linear speedup `1.02x`;
    - `v_proj`: base unique `383`, output MSE `5.97e-3`, local bpw `5.03`, decode speedup `1.03x`, decode-plus-linear speedup `1.02x`;
    - `o_proj`: base unique `1424`, output MSE `7.62e-2`, local bpw `5.03`, decode speedup `1.05x`, decode-plus-linear speedup `1.02x`;
    - `gate_proj`: base unique `611`, output MSE `8.46e-3`, local bpw `5.03`, decode speedup `1.03x`, decode-plus-linear speedup `1.02x`;
    - `up_proj`: base unique `425`, output MSE `2.99e-4`, local bpw `5.03`, decode speedup `1.04x`, decode-plus-linear speedup `1.02x`;
    - `down_proj`: base unique `1317`, output MSE `4.14e-2`, local bpw `5.03`, decode speedup `1.04x`, decode-plus-linear speedup `1.01x`.
- Reading these numbers:
    - the idea transfers well to `q_proj` and especially `up_proj`;
    - it is usable but clearly rough on `k_proj`, `v_proj`, and `gate_proj`;
    - it is currently not good enough for `o_proj` and `down_proj`, whose selected tiles still have very high local route diversity.
- Speed interpretation must stay honest. The storage/traffic proxy improved sharply (`11 bpw` to about `4.0-5.0 bpw`), but the current naive Torch realization only yields about `1.01x-1.07x` runtime speedup on the tile microbench. This confirms that better structure alone is not enough; the next systems step still has to be a dedicated local-palette kernel.

### First real local-palette Triton kernel

- `src/triton_local_palette_matmul.py` now provides a first Triton matmul path that uses tile-local palette indices instead of global route IDs.
- `experiments/m6e_local_palette_kernel_bench.py` benchmarks this kernel on selected layer-0 tiles for `q_proj`, `up_proj`, `o_proj`, and `down_proj`, across palette sizes `16`, `32`, and `64`.
- The central systems result is that the local kernel is materially better than the global-ID Triton path:
    - `q_proj`: `1.23x-1.34x` local-vs-global Triton speedup;
    - `up_proj`: `1.24x-1.27x`;
    - `o_proj`: `1.26x-1.39x`;
    - `down_proj`: `1.23x-1.25x`.
- Just as important, this speedup is fairly flat across palette size `16/32/64`, while quality improves as palette size grows. That means the local-palette execution path is not obviously punished by moving from `16` to `32` or `64` local values.
- However, dense still wins on absolute latency. The current local kernel reaches only about `0.32-0.37` of dense speed in the tile microbench, so it is still roughly `2.7x-3.1x` slower than dense. The new result is therefore not "runtime solved"; it is "the right execution direction is finally visible in code and benchmarks".

### Depth transfer on `q_proj` and `up_proj`

- `experiments/m6d_route_distill_depth_sweep.py` extends the analysis to `q_proj` and `up_proj` over layers `0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79`, using palette `32` on the top sampled high-diversity tile of each layer.
- Aggregate results:
    - `q_proj`: mean base unique `1281.6`, mean distilled output MSE `3.41e-2`, mean local bpw `5.03`, mean local-vs-global Triton `1.26x`, mean local-vs-dense `0.337`;
    - `up_proj`: mean base unique `1287.7`, mean distilled output MSE `4.06e-2`, mean local bpw `5.03`, mean local-vs-global Triton `1.26x`, mean local-vs-dense `0.335`.
- Best layers in this sweep are still the shallow ones: `q_proj` layer `0` gave `9.99e-4`, and `up_proj` layer `0` gave `6.10e-4` on the selected tile.
- Worst layers are much worse: `q_proj` layer `16` reached about `5.20e-2`, and `up_proj` layer `48` about `6.82e-2`.
- This is an important split conclusion:
    - the local kernel speedup is stable across depth;
    - the fixed palette-32 student quality is not stable across depth on top-diversity tiles.
- So the next quality question is no longer whether local palette compression works at all, but where to deploy it selectively and whether harder families need a larger palette, a better objective, or an escape hatch.

### First selective rollout policy

- `experiments/m6f_selective_runtime_policy.py` converts the depth-sweep into an explicit deployment policy.
- Current thresholds are deliberately conservative: selected-tile output MSE must be at most `1e-3`, and local-vs-global Triton speedup must be at least `1.2x`.
- Under these rules the policy is narrow: only `model.layers.0.self_attn.q_proj.weight` and `model.layers.0.mlp.up_proj.weight` are approved for local palette execution; the remaining 20 checked `q_proj/up_proj` candidates fall back to global-ID Triton.
- This is a useful result, not a disappointment. It means the runtime path is now tied to an explicit safety rule instead of intuition.

### First corrected full-layer tiled runtime benchmark

- `experiments/m6g_full_layer_tiled_runtime_bench.py` benchmarks the approved layers under a real full-layer tiled execution path.
- During this step a real kernel bug was found and fixed: the autotuned Triton launch grid had been computed using hardcoded `32 x 32` geometry instead of the selected `BLOCK_M/BLOCK_N`, which could corrupt tiled outputs. After switching to meta-aware grids, the full-layer local path became numerically correct again.
- Corrected layer-0 full-layer results:
    - `q_proj`: dense `0.102 ms`, global full Triton `1.45 ms`, global tiled `615.1 ms`, local tiled `575.7 ms`, local-vs-global tiled `1.07x`, output MSE against routed dense `1.09e-5`;
    - `up_proj`: dense `0.364 ms`, global full Triton `4.88 ms`, global tiled `2190.3 ms`, local tiled `1996.7 ms`, local-vs-global tiled `1.10x`, output MSE `4.77e-6`.
- This changes the runtime interpretation in an important way:
    - the tile-local execution idea is now correct and slightly better than global tiled execution;
    - but tile launch overhead completely dominates the full-layer runtime, so both tiled paths are still orders of magnitude slower than the non-tiled fused global kernel.
- Therefore the next systems milestone is now sharply defined: fuse or group many tiles into a single launch regime. Without that, the local palette advantage remains trapped inside tile microbenchmarks.

### Grouped multi-tile execution

- `src/full_layer_tiled_runtime.py` now supports grouped local execution over wider `group_cols`, and `experiments/m6h_grouped_local_runtime_bench.py` sweeps this grouping width on the policy-approved layers.
- This directly tests whether removing launch granularity is enough to recover most of the lost runtime.
- The answer is positive, but incomplete.
- For `layer0.q_proj`:
    - `group_cols=128`: `4096` launches, grouped-local `579.5 ms`, grouped-global `614.8 ms`;
    - `group_cols=2048`: `256` launches, grouped-local `34.6 ms`, grouped-global `38.6 ms`;
    - `group_cols=8192`: `64` launches, grouped-local `15.3 ms`, grouped-global `15.2 ms`.
- For `layer0.up_proj`:
    - `group_cols=128`: `14336` launches, grouped-local `1922.7 ms`, grouped-global `2168.3 ms`;
    - `group_cols=2048`: `896` launches, grouped-local `65.1 ms`, grouped-global `73.0 ms`;
    - `group_cols=8192`: `224` launches, grouped-local `49.5 ms`, grouped-global `52.1 ms`.
- Output accuracy remains essentially exact across the sweep: grouped-local output MSE stays around `1e-5` to `1e-6` against the routed dense reference.
- This is the strongest runtime result so far because it separates two bottlenecks:
    - launch overhead was indeed a major part of the tiled-runtime collapse;
    - but even after grouping, the best local path remains about `10x` slower than the fused non-tiled global kernel.
- There is also one useful anomaly: `up_proj` at `group_cols=512` made grouped-local slower than grouped-global. That suggests the next runtime frontier is not only fewer launches, but also better kernel efficiency across palette-size/group-shape regimes.
- Practical reading:
    - `group_cols=8192` is the best pure-latency grouped regime tested so far;
    - `group_cols=2048` is the best current compromise if we want both a large launch reduction and a still-visible local advantage over grouped-global execution.

### 2D grouped execution over rows and columns

- `experiments/m6i_grouped_2d_runtime_bench.py` extends the grouped benchmark to sweep both `group_rows` and `group_cols` on the policy-approved layers.
- This tests the next narrow runtime hypothesis: after grouping columns, can we reduce launch count further by grouping more output rows into the same local execution block?
- The answer is yes.
- For `layer0.q_proj`:
    - `(group_rows=128, group_cols=2048)`: `256` launches, grouped-local `34.37 ms`, grouped-global `38.13 ms`;
    - `(256, 2048)`: `128` launches, grouped-local `18.05 ms`;
    - `(512, 2048)`: `64` launches, grouped-local `9.45 ms`;
    - `(512, 8192)`: `16` launches, grouped-local `5.33 ms`, grouped-global `5.56 ms`.
- For `layer0.up_proj`:
    - `(128, 2048)`: `896` launches, grouped-local `125.95 ms`, grouped-global `140.12 ms`;
    - `(256, 2048)`: `448` launches, grouped-local `62.89 ms`;
    - `(512, 2048)`: `224` launches, grouped-local `31.41 ms`;
    - `(512, 8192)`: `56` launches, grouped-local `15.70 ms`, grouped-global `17.02 ms`.
- Output accuracy remains effectively unchanged: grouped-local output MSE stays around `1.14e-5` / `4.58e-6` for `q_proj` and `4.86e-6` / `1.37e-6` for `up_proj`, depending on `group_cols`.
- This moves the absolute runtime gap materially:
    - `q_proj`: best grouped-local falls from `15.3 ms` in the 1D grouped case to `5.33 ms`, versus `1.44 ms` for fused global full Triton;
    - `up_proj`: best grouped-local falls from `49.5 ms` to `15.70 ms`, versus `4.88 ms` for fused global full Triton.
- Two regimes stand out now:
    - best pure-latency point: `(group_rows=512, group_cols=8192)`;
    - best compromise with a still-clear local edge over grouped-global: `(group_rows=256, group_cols=8192)`, where local-vs-grouped-global reaches `1.17x` on `q_proj` and `1.25x` on `up_proj`.
- So the runtime frontier has narrowed again. The main remaining bottleneck is no longer just launch granularity, but the arithmetic efficiency of the local kernel as the 2D group grows and the local palette union becomes larger.

### Larger 2D shapes: the frontier keeps moving

- `experiments/m6j_grouped_shape_frontier_bench.py` extends the 2D sweep further to `group_rows=1024` and `2048`, and also logs both mean and total local palette-union size.
- This tests a sharper question: after the M6i win, do larger row groups still help, or do they collapse once the local palette union becomes too large?
- On the approved tensors, larger row groups still help clearly.
- For `layer0.q_proj`:
    - `(256, 8192)`: `32` launches, grouped-local `8.74 ms`, total local union `46828`;
    - `(512, 8192)`: `16` launches, grouped-local `5.15 ms`, total local union `24791`;
    - `(1024, 8192)`: `8` launches, grouped-local `3.17 ms`, total local union `12658`;
    - `(2048, 8192)`: `4` launches, grouped-local `1.82 ms`, total local union `6370`.
- For `layer0.up_proj`:
    - `(256, 8192)`: `112` launches, grouped-local `26.63 ms`, total local union `140824`;
    - `(512, 8192)`: `56` launches, grouped-local `16.11 ms`, total local union `78579`;
    - `(1024, 8192)`: `28` launches, grouped-local `11.09 ms`, total local union `41714`;
    - `(2048, 8192)`: `14` launches, grouped-local `6.44 ms`, total local union `21315`.
- Accuracy remains essentially unchanged across the sweep: grouped-local output MSE stays around `1.20e-5` / `4.58e-6` on `q_proj` and `4.98e-6` / `1.36e-6` on `up_proj`, depending on `group_cols`.
- The best pure-latency point is now `(group_rows=2048, group_cols=8192)`:
    - `q_proj`: `1.82 ms` local vs `1.44 ms` fused global full Triton;
    - `up_proj`: `6.44 ms` local vs `4.94 ms` fused global full Triton.
- So on the currently approved layers, the local path is now within roughly `1.27x-1.30x` of fused global full Triton.
- A more conservative near-frontier point is `(1024, 8192)`, where local still beats grouped-global (`1.04x-1.05x`) and reaches `3.17 ms` on `q_proj` and `11.09 ms` on `up_proj`.
- The union statistics are important for interpretation:
    - mean union per group rises only modestly as groups widen;
    - total union over the full layer falls sharply because the number of launches collapses faster than per-group union grows.
- This changes the bottleneck reading again. The remaining gap is no longer well explained by local-value diversity alone; it points more directly to arithmetic efficiency and scheduling quality of the grouped local kernel at very large 2D blocks.

### Index bandwidth still matters at the frontier

- A short additional sanity gate on `100` raw WikiText-2 windows confirms that the quality picture has not changed: dense `3.2749`, routed `3.2773`, gap `0.00238`.
- The same `100`-window run also gives a simple end-to-end eval throughput reading: `1006.4 tok/s` dense versus `1016.1 tok/s` routed over `52736` tokens. This is not a decode benchmark, but it confirms that the routed model is not showing an obvious full-model regression on this short text gate.
- On the runtime side, one negative and one positive systems result followed immediately after M6j.
- Negative result: expanding the local Triton autotune config set with wider block shapes changed the frontier runtime by less than `0.1%`. So the remaining gap was not explained by an obviously missing block configuration.
- Positive result: local palette indices were still stored as `int32` even though the measured local unions stayed far below the `int16` limit.
- Switching `build_local_palette_repr` to emit `int16` local indices materially improved frontier performance:
    - `q_proj`, `(1024,8192)`: `3.17 ms -> 2.71 ms`;
    - `q_proj`, `(2048,8192)`: `1.82 ms -> 1.62 ms`;
    - `up_proj`, `(1024,8192)`: `11.09 ms -> 9.14 ms`;
    - `up_proj`, `(2048,8192)`: `6.44 ms -> 5.62 ms`.
- That is a consistent `11-18%` speedup with no visible correctness cost: output MSE remains around `4.86e-6` on `q_proj` and `1.33e-6` on `up_proj` for the checked shapes.
- The interpretation is sharper now:
    - very large 2D grouping already solved most of the launch problem;
    - raw index bandwidth was still a real part of the remaining local-kernel overhead;
    - after halving that index bandwidth, the local path is within about `1.13x` of fused global full Triton on `q_proj` and `1.05x` on `up_proj` at `(2048,8192)`.
- So the remaining systems gap is now extremely narrow. The next likely gain must come from how palette values themselves are staged and reused inside the kernel, not from broader grouping alone.

### FP16 palette storage: helpful, but not uniformly

- The next nearby probe was to halve palette-value bandwidth as well, by storing the local palette itself in `fp16` rather than `fp32`.
- This was checked on the same frontier shapes and rerun once more to verify that the sign was stable.
- The effect is real, but shape-dependent:
    - `q_proj`, `(1024,8192)`: `2.71 ms -> 2.48 ms` (`-8.6%`);
    - `q_proj`, `(2048,8192)`: `1.62 ms -> 1.57 ms` (`-2.8%`);
    - `up_proj`, `(1024,8192)`: `9.14 ms -> 8.44 ms` (`-7.7%`);
    - `up_proj`, `(2048,8192)`: `5.62 ms -> 5.89 ms` (`+4.9%`).
- So `fp16` palette storage improves 3 of the 4 checked frontier shapes, but slightly hurts the largest `up_proj` case.
- Importantly, correctness remains unchanged for practical purposes: grouped-local output MSE stays around `4.24e-6` on `q_proj` and `1.35e-6` on `up_proj`.
- The interpretation is now quite specific:
    - index bandwidth clearly mattered and was an easy win;
    - palette-value bandwidth matters too, but interacts with shape and kernel behavior in a less uniform way;
    - the final systems step is unlikely to be another trivial dtype change. It will more likely come from explicit palette staging/reuse inside the grouped local kernel.

### Hot-prefix usage is real; naive hot-prefix execution is not

- A follow-up profile (`experiments/m6o_palette_hotness_profile.py`) measured how concentrated the local palette usage actually is on the current frontier groups.
- For `q_proj`, the concentration is strong:
    - `top8 ≈ 0.677`, `top16 ≈ 0.817`, `top32 ≈ 0.908`, `top64 ≈ 0.957`;
    - only about `30` entries are needed on average to cover `90%` of local-index usage.
- For `up_proj`, the distribution is flatter but still not uniform:
    - `top8 ≈ 0.272`, `top16 ≈ 0.434`, `top32 ≈ 0.626`, `top64 ≈ 0.798`;
    - about `105` entries cover `90%` of usage.
- This made a hot-prefix staged local kernel look plausible in principle, so a direct probe was added in `src/triton_local_palette_hotprefix_matmul.py` and benchmarked via `experiments/m6p_hotprefix_frontier_bench.py`.
- The result is an important negative one.
- With `hot_size=32`, the prototype is dramatically slower than the baseline grouped-local path:
    - `q_proj`: `2.48 ms -> 6.14 ms` at `(1024,8192)` and `1.57 ms -> 5.48 ms` at `(2048,8192)`;
    - `up_proj`: `8.46 ms -> 21.45 ms` and `5.93 ms -> 19.09 ms`.
- Reducing the prefix to `hot_size=16` improves the prototype a lot, but still not enough:
    - `q_proj`: `2.99 ms` and `2.46 ms`;
    - `up_proj`: `10.47 ms` and `8.61 ms`.
- So `hot16` is roughly `2x` faster than `hot32`, but still `21%-57%` slower than the normal local grouped kernel.
    - A final nearby check at `hot_size=8` improves the naive path again, but still does not flip the sign:
        - `q_proj`: `2.486 ms` vs `2.479 ms` baseline at `(1024,8192)`, and `1.800 ms` vs `1.574 ms` at `(2048,8192)`;
        - `up_proj`: `8.657 ms` vs `8.429 ms`, and `6.257 ms` vs `5.861 ms`.
    - Re-running `hot8` with `2048` tokens makes the conclusion stronger rather than weaker: the naive hot-prefix kernel falls to only about `0.59x-0.62x` of the baseline grouped-local throughput.
    - The interpretation is now sharper still:
    - the usage profile says a hot prefix exists, especially for `q_proj`;
    - but a naive per-element unrolled `where` chain is the wrong way to exploit it;
        - and after the `hot8` plus large-`M` checks, the current naive hot-prefix branch should be treated as closed;
        - if this idea is pursued further, it needs actual staged/shared hot-prefix handling rather than branch-heavy scalar substitution inside the inner loop.

    ### Large row groups were not saturated after all

    - The next systems check revisited a local assumption that had quietly become stale: after the `int16` and `fp16` representation changes, was `(2048,8192)` still the real frontier, or just the frontier we had last measured?
    - It turned out to be the latter. Larger row groups continue to help materially on the current local-path representation.
    - On `layer0.q_proj`:
        - `(4096,8192)` reaches about `0.97 ms`;
        - full-row `(8192,8192)` stabilises around `0.95 ms` across reruns.
    - On `layer0.up_proj`:
        - `(4096,8192)` reaches about `3.40 ms`;
        - `(8192,8192)` stays near `3.40 ms`;
        - `(16384,8192)` improves further to `3.26 ms`;
        - full-row `(28672,8192)` reaches `3.13 ms`.
    - These are no longer ``almost there'' numbers relative to fused global full Triton. On the currently approved tensors, grouped-local is now actually faster:
        - `q_proj (8192,8192)`: about `1.52x` faster than fused global full;
        - `up_proj (28672,8192)`: about `1.56x` faster than fused global full.
    - `experiments/m6s_shape_runtime_policy.py` turns that benchmark frontier into an explicit deployment artefact by selecting the fastest valid grouped-local shape per approved tensor.
    - The current shape-aware runtime policy therefore picks:
        - `model.layers.0.self_attn.q_proj.weight -> (8192,8192)`;
        - `model.layers.0.mlp.up_proj.weight -> (28672,8192)`.
    - This changes the immediate engineering priority. The next step is no longer ``find one more micro-optimisation inside the naive hot-prefix probe''; it is to wire this shape-aware grouped-local dispatch into the runtime path and then widen the set of quality-approved tensors.

    ### First full-model selective-runtime gate: speed measured, fused rollout still unstable

    - `experiments/m6t_selective_runtime_gate.py` is the first short full-model gate that actually uses the new dispatch policy rather than stopping at per-layer microbenchmarks.
    - On a `16`-window raw WikiText-2 run, it replaces all `560` targeted linears with runtime modules:
        - `2` layers use grouped-local dispatch from the shape policy;
        - `558` layers use fused global-ID Triton.
    - This gives the first end-to-end throughput number for the selective-runtime stack itself: about `23.78 tok/s` over `9728` evaluated tokens.
    - The result is useful but not yet deployable, because routed perplexity becomes `NaN` under that wide fused rollout.
    - Two nearby isolation checks sharpen the interpretation:
        - replacing only the two grouped-local approved layers stays fully finite on a one-chunk full-model pass;
        - replacing all target layers with the safe materialize path (`PackedIDRouteLinear`) also stays finite on a one-chunk pass.
    - So the current blocker is no longer whether shape-aware grouped-local dispatch works. It does. The blocker is numerical stability of the broad fused-global runtime path when rolled out across the model.
    - That would still leave open one practical question: even if fused rollout is unstable, what is the best deployable selective runtime today?
    - A direct control run answers that cleanly. On the same short gate, `packed + grouped-local policy` is already both correct and much faster than the wide fused rollout:
        - `4` windows: `PPL = 2.3755`, `351.32 tok/s`;
        - runtime mix: `558` packed layers plus the same `2` grouped-local policy layers.
    - This motivated a different runtime direction entirely: not another fused-only rescue, but a bounded cached-packed path.
    - In that variant, route weights are materialised lazily and cached only for layers whose dense weight tensor fits under a per-layer cache budget; larger layers remain on the normal packed path.
    - With a `128 MB` cache cap, this becomes the strongest current deployment candidate:
        - `4` windows: `PPL = 2.3745`, `367.89 tok/s`;
        - `16` windows / `9728` tokens: `PPL = 2.7798`, `404.55 tok/s` in `24.05 s`.
    - The cache audit is also informative. Over the `16`-window gate, `558` layers are under cached-packed control, producing `319` cache misses, `4785` cache hits, and `3824` deliberate cache skips for layers that exceed the memory cap.
    - This changes the runtime strategy substantially:
        - grouped-local stays as a shape-aware override where it wins clearly;
        - bounded cached-packed becomes the safe default path for the rest of the model;
        - fused global-ID Triton is now an experimental promotion path that must earn its way back through explicit layer validation rather than by being the default rollout.
    - After fixing a bug in runtime-kwargs propagation, that promotion path was measured properly rather than only sketched. A real adaptive-shadow gate with `4` validation calls shows that only `180` of the `558` fused-global candidates survive layer-wise validation, while `377` layers trigger packed-reference mismatches and one more layer hits a non-finite fallback.
    - `experiments/m6u_fused_promotion_policy.py` turns that audit into a fused allowlist, and `replace_with_hybrid_runtime` uses it to build a hybrid execution path:
        - grouped-local on the two policy-approved tensors;
        - fused global-ID only on the validated allowlist;
        - bounded cached-packed on the remaining layers.
    - This hybrid path is fully finite, but it is not a performance win:
        - `4` windows: `PPL = 2.3760`, `171.66 tok/s`;
        - `16` windows / `9728` tokens: `PPL = 2.7789`, `204.20 tok/s`.
    - That is dramatically slower than bounded cached-packed alone (`367.89 tok/s` and `404.55 tok/s` respectively). So the fused-promotion allowlist is currently a paper-worthy negative result, not a new deployment candidate.
    - The present runtime conclusion therefore becomes sharper still:
        - bounded cached-packed plus grouped-local shape policy is the best current deployable stack;
        - fused promotion remains an experimental branch that still loses end-to-end, even after allowlisting only the layers that pass shadow validation.
    - A separate comparison gate now makes that deployment tradeoff explicit in one place instead of across multiple experiments: `experiments/m6v_baseline_vs_deployment_gate.py` evaluates the untouched baseline model and the current deployment runtime on the same windows, and reports `PPL`, `tok/s`, VRAM, layer count, and runtime mix.
    - On the current `16`-window raw WikiText-2 run:
        - baseline: `PPL = 2.7805`, `1106.28 tok/s`, max peak VRAM `47214.8 MB`;
        - deployment (`2` grouped-local + `558` bounded cached-packed): `PPL = 2.7803`, `409.47 tok/s`, max peak VRAM `102166.9 MB`, replacement-stage peak `102916.6 MB`.
    - This is the cleanest current statement of where the project stands operationally:
        - quality is effectively preserved on this gate;
        - runtime is still only about `37%` of baseline throughput;
        - VRAM cost is substantially higher under the deployable route runtime.

### 4. Hybrid route base plus small residual branch

This is a fallback if tile-local grouping alone is not enough.

- Keep the route representation for most weights, but add a tiny correction branch only for hard tiles or hard layers.
- That residual could be low-rank, sparse, or selectively dense.
- This is less elegant than a pure route runtime, but it may be a good compromise if the last quality or speed gap is concentrated in a small fraction of the model.

## Measurable artefacts per run

- Ladders: `ladders.pt` per family.
- Per-matrix routes: `routes/*.pt` (intermediate, can be discarded after codebook build).
- Codebook: `codebook.pt` (unique routes + packed digits + stop_depth).
- Final weights: `ids.pt` (one integer tensor per original linear layer).
- Manifest: `manifest.json` with bpw breakdown (digits + stop_depth + IDs + codebook overhead + row_scale).

## First-cut bpw estimate (rough)

- Digits: avg 6 digits × log2(3) ≈ **9.5 bpw** before codebook (pre-ID form).
- Stop depth: 4 bits per weight = **4 bpw**.
- With ID codebook and typical redundancy, expect effective form dominated by `ceil(log2 M_f)` bpw. If `M_f ≈ 2^16`, that is 16 bpw for the ID tensor *minus* whatever storage the digits would have cost per weight (they now live once in the codebook).

The point: bpw is a reporting metric, not a target. The target is dense-level PPL.

## Current runtime split (2026-04-19)

The project now has two runtime stories, and mixing them leads to bad decisions.

- **Operational / deployable branch.** This is the materialized runtime family used to answer: what is the best full-model path today if we want strong quality and real throughput?
- **Packed research branch.** This is the exact Block-RVQ runtime family used to answer: can the discrete language itself be executed faster without changing the encoding?

After fixing the compare harness, the current `16`-window raw WikiText-2 picture is:

- baseline: `2.7805` PPL, `1169.63 tok/s`, peak `47214.8 MB`;
- eager-bf16: `2.7775` PPL, `1240.29 tok/s`, peak `47213.8 MB`;
- eager-hybrid: `2.8527` PPL, `1402.63 tok/s`, peak `33492.5 MB`.

So the current deployment reading is simple:

- if we want baseline-level quality, `eager-bf16` is the best operational path so far;
- if we accept a quality tradeoff, `eager-hybrid` is the current speed/VRAM frontier;
- the packed runtime remains a research track, not the present deployment default.

## M20-M24 changed the question

The recent Block-RVQ runtime work did not just add optimizations; it changed what the real question is.

- **M20 fast reconstruct** established a fair packed baseline. On an `8192 x 8192` microbench, `full_weight_fast` reduced the old `full_weight` path from `33.74 ms` to `24.95 ms`, and `per_group_fast` reduced `35.03 ms` to `26.56 ms`, with effectively unchanged relMSE.
- **M21/M22/M24 stage-control** showed that "use fewer residual stages" is a real but blunt lever. On `first8_qk_gu`, the conservative per-layer `cos >= 0.999` policy gave `2.9614` PPL and `391.47 tok/s` versus `2.9487` / `385.47` for the 3-stage baseline, while uniform 2-stage (`3.4761` / `497.58`) and `cos >= 0.99` (`3.5730` / `408.85`) were clearly dead.

This reframed the runtime problem. The main question is no longer just "how many stages should we keep?" It is "which `(stage, id)` tokens matter enough to keep hot?"

## Grammar-aware packed runtime

M23 introduced an activation-weighted discrete-language view of the packed model. For each `(stage, id)` token we accumulate influence mass from the activation norm, row scale, and codebook vector norm.

The key result is split-level concentration.

- Merged layer-wide vocabularies are too diffuse to justify a naive per-layer hot cache.
- On `first8_qk_gu`, average merged `top32_share` is only `0.058` for attention and `0.066` for MLP.
- But individual sub-stages are much more concentrated: average stage `top8_share` is `0.107`, median is `0.101`, and the maximum observed value is `0.451`.

That means the useful grammar signal lives at the stage-local level, not in one merged layer vocabulary.

This directly motivated `full_weight_hot`, which performs a stage-local hot/cold split. Once M25 added persisted encodings and same-encoding evaluation, the result became clear:

- `l0_qkv_gu`, 4 windows: `1228.23 -> 1335.39 tok/s` at identical `PPL = 56.9796`;
- `l54_q_gu`, 4 windows: `1308.03 -> 1396.46 tok/s` at identical `PPL = 2.3850`;
- `l54_q_gu`, 16 windows: `933.41 -> 962.71 tok/s` at identical `PPL = 2.7996` and the same peak memory.

This is the first clean proof that grammar-aware packed execution can be an exact runtime win when measured fairly.

## Why same-encoding evaluation is mandatory

Independent re-encodes can easily swamp the runtime effect we are trying to measure. Even with the same high-level RVQ configuration, different encode passes introduce enough noise to hide or invert small throughput wins.

That is why the packed branch now depends on:

- persisted encoding I/O in `src/encoding_io.py`;
- preencoded runtime replacement;
- same-process / same-encoding strategy comparison.

Without that methodology, we do not know whether we measured runtime or re-encoding drift.

## Important negative result: fused Triton is still dead here

The Triton Block-RVQ path now supports calibrated stage scales, so it can be compared fairly on the same encoded weights. That removed a tooling blocker, but not the performance blocker.

- On `l54.q_proj` microbench, the same-encoding Triton path is still about `45.4 ms` versus `24.6 ms` for `full_weight_fast`.
- On `l0_qkv_gu`, 4 windows, `triton_block_rvq` reaches only `534.05 tok/s` versus `1228.23 tok/s` for `full_weight_fast`.

So the current runtime interpretation is sharp:

- the stage-local grammar signal is real;
- Python-level exact hot/cold already helps;
- the present fused Triton path should still be treated as a negative result for this workload.

The more recent native stage-local probes do not change that conclusion yet. Probe B0 (`stage_local_hot_cold`) fixed the representation surface and passed one-hot parity, but stayed slower than `full_weight_hot_v2` and still failed the honest full harness. Probe B1 (`stage_local_hot_cold_b1`) removed the explicit `hot_lut` indirection and still kept one-hot parity plus `has_nan = False`, but realistic `m = 2048` latency collapsed from about `58.8 ms` to `569.4 ms` on both `l54.gate_proj` and `l54.up_proj`. So the current Triton direction is not "almost there"; it is still structurally doing too much work inside the inner loop.

## Next concrete step

The next systems milestone is no longer "try another broad packed runtime idea." It is narrower:

- keep `eager-bf16` as the operational quality-preserving baseline;
- keep `eager-hybrid` as the current speed/VRAM deployment frontier;
- for the packed branch, redesign stage 2 as a true stage-local hot-palette kernel that stages each stage hot set once per block/tile, and evaluate it only under the same-encoding harness.

## Weight Assembly Language (WAL) framing

The project is now framed as building the first weight-assembly toolchain rather than another quantizer.

- A weight `w` is treated as a machine instruction in a per-model neural ISA.
- Block-RVQ encoding is the disassembler `w -> (stage, id) tokens`.
- The codebook and ladder are the register file and immediate values.
- Stage-local hot/cold is the first compiler optimisation (`-O1`).
- The same-encoding harness is the reproducible build environment for weight programs.

Key formal objects that will appear in every following milestone:

- Influence mass per token at position `(n,k)`:

$$ I_{l,n,k}(s,i) = \| x_{l,n} \|_2 \cdot r_{l,n} \cdot \| c_{l,s,i} \|_2 \cdot |d_{s,i}| $$

- Stage-local grammar entropy with $p(i) \propto I_{l,s}(i)$:

$$ H_{l,s} = - \sum_{i \in V_{l,s}} p(i) \log p(i) $$

- Reconstruction error budget:

$$ \mathbb{E}[\| W - \hat W^{(S)} \|_F^2] \le \sum_{s=1}^{S} \lambda_s \sigma_s^2 $$

- Deployment quality:

$$ \text{WAL-Q} = \alpha (1 - \Delta\text{PPL}) + \beta \frac{\text{tok/s}_{\text{WAL}}}{\text{tok/s}_{\text{dense}}} + \gamma \frac{\text{VRAM}_{\text{dense}}}{\text{VRAM}_{\text{WAL}}} $$

## Component-specific WAL dialects

Llama-3.3-70B does not have one weight language. Different components want different dialects:

- `WAL-Attn` for `q/k/v/o`: needs RoPE-phase aware influence weighting (early heads use phase-coherent grammar).
- `WAL-MLP` for `gate/up/down`: needs SwiGLU-aware influence; `up` and `gate` compress easily, `down` is the consistently hard tensor.
- `WAL-Head` for `lm_head`: needs vocabulary-aware codebook.
- `WAL-Norm` for RMSNorm scales: a small ternary or 8-bit scale+shift dialect.

These dialects share the same `(stage, id)` ISA but diverge in influence weighting and hot/cold policy.

## Roadmap M26-M30 in WAL terms

- **M26**: fused exact hot/cold runtime. Stage 1 is allocation-free Torch (`full_weight_hot_v2`) and remains the last honest packed positive point. Stage-2 probe A (tile-local persistent palette), probe B0 (native scaffold), B1 (direct hot-id hot palette), B2 (true staged hot palette), and B3 (selection-only fix on top of B2) are all negative as frontier candidates. The later RRF and PTDP probes showed that the current `12 x 256` language is too diffuse for small cached vocabularies. FGRL Step 3a then tested the stronger ISA-change hypothesis directly by moving to a total `20 x 80` encoding. That move improves quality (`avg_rel_mse 0.0390 -> 0.0172`, `PPL 2.4054 -> 2.3928`) and pushes tile `top64` share to about `0.91`, but it still does not create the desired sparsity: normalized tile occupancy rises to `~0.996`, meaning almost the whole 80-id vocabulary is still live per tile. WAL-SS Step 4a then tested the orthogonal syntax hypothesis directly: keep the current encoding exact, but impose a lossless macro layer over stage-id programs. That also preserves quality perfectly, yet reveals almost no reusable syntax because the block-program stream is already almost fully unique. WAL-HP Step 5a strengthened that syntax hypothesis by moving to shared cross-layer subroutines over `l53/l54 gate/up`; it found real reuse in the narrow sense (`136 / 256` selected subroutines are shared across at least two targets), but the effect is still trace-level because global `call_coverage` is only `0.000166` and average program length remains `11.9973-11.9992` out of raw `12`. WAL-LRT Step 6a finally let the language create its own discrete reusable surface through learned templates on `l54.gate/up`; that raises exact syntax coverage to about `0.041` and shortens average program length to about `11.75 / 12`, but the nearest-template semantics are still too weak (`template_only` relMSE about `1.81`, `PPL 2.4081 -> 2.9375`). WAL-FG Step 7a then imposed an explicit shallow formal grammar with `15` rules over four `3`-token phrase slots; that does give every block program a real depth-`4` parse tree, but the exact parse is still almost entirely literal (`11.9984 / 11.9988` out of `12`, rule coverage `0.000272 / 0.000203`), while `grammar_only` still degrades to relMSE about `1.31` and `PPL 2.4081 -> 2.6866`. WAL-TS Step 8a then added explicit route types over the same slice (`MLP_GATE/UP x COMMON/MIXED/OUTLIER`), which does separate common and outlier blocks slightly, but the exact typed grammar remains almost entirely literal (`11.99815 / 11.99848` out of `12`, typed rule coverage `0.000308 / 0.000253`) and `typed_only` degrades even more strongly (`eval-layer relMSE about 1.75`, `PPL 2.4081 -> 2.7994`). WAL-ASM Step 9a then turned the same learned full-program patterns into a classical assembly surface with `64` labeled subroutines, `32` macros, dominant `JMP` targets, and exact `CALL + literal corrections`; that does create an explicit assembly library, but the structure is still trace-level (`11.9288 / 11.9309` out of `12`, `avg_calls ~ 0.07`, `avg_jumps ~ 1.7e-4`, `call_token_coverage ~ 0.011`, `macro_body_coverage ~ 0.072-0.075`), while `asm_only` degrades even more strongly (`eval-layer relMSE about 1.85`, `PPL 2.4081 -> 2.9601`). WAL-LDI Step 10a then introduced the first explicitly designed two-level semantic ISA on the same slice: `4` learned high-level families plus `64` family-conditioned low-level atoms per layer. This is the first probe that creates a real semantic partition rather than only a wrapper: all four families are active, family entropy is about `1.14`, and a small `COMMON_CORE` cluster separates strongly with margin `~6.3 / 6.1`. But the actual program surface is still not usable: exact high-level program length grows to `12.9969 / 12.9971` out of raw `12`, low-level atoms fire only `~0.003` times per block, token coverage is only `~0.0005`, and `ldi_only` still degrades to `eval-layer relMSE ~1.77` and `PPL 2.4081 -> 2.8318`. WAL-E2E Step 11a then moved the semantic-ISA idea into the encoder loop itself: a small semantic encoder plus family-conditioned atom bank are trained jointly on sampled block programs with token reconstruction and semantic regularization. This does keep all four families alive in the authoritative run and preserves a coherent `COMMON_CORE` island, so the semantic partition is no longer only post-hoc. But the actual program still does not become usable: exact program length rises to `12.9988 / 12.9988`, low-level atom usage drops further to about `0.00119 / 0.00118`, token coverage falls to about `0.000199 / 0.000197`, and `e2e_only` degrades to `eval-layer relMSE ~1.70` and `PPL 2.4081 -> 2.8847`. WAL-DR Step 12a then changed the controlling objective itself: the same semantic encoder and family-conditioned atom surface are now trained against direct final-block reconstruction plus an explicit program-cost proxy instead of token CE. That shift is already meaningful. All four families remain active (`family_entropy ~1.12 / 1.25`), and the approximate `dr_only` surface improves materially over both WAL-LDI and WAL-E2E (`eval-layer relMSE ~1.54`, `PPL 2.4081 -> 2.7325`). But the exact program is still almost entirely correction-dominated: average program length remains `12.9984 / 12.9988`, low-level atoms still fire only `0.00156 / 0.00117` times per block, and token coverage remains only `0.000260 / 0.000195`. WAL-LO Step 12b then moved the loss one step closer to true downstream behavior by reconstructing activation-conditioned layer-output contributions on real captured layer inputs and by adding stronger exact-program cost plus an explicit atom-vs-literal decision head. This still does not create a usable exact ISA. Exact quality remains identical by construction, the approximate `lo_only` path is slightly worse than WAL-DR (`PPL 2.4081 -> 2.7463`), and the exact program almost completely collapses to literal fallback: average program length is `13.0018 / 13.0000`, low-level calls only `0.0140 / 0.0000`, and low-level token coverage only `0.00102 / 0.00000`. WAL-LHA Step 12c then changed the atom surface itself: atoms become tiny expressive modules that generate context-conditioned token logits on top of a learned base phrase prior, with a separate atom-selection loss and a direct atom-usage bonus. This does produce the best approximate surface so far in the objective line (`PPL 2.4081 -> 2.7179`, slightly better than WAL-DR and better than WAL-LO), and sample output-relMSE improves materially to about `1.34 / 1.21`. But the exact program still does not become economically useful: average program length remains `13.0006 / 13.0000`, low-level calls only `0.0083 / 0.0000`, and exact fallback remains overwhelmingly literal. WAL-SBC Step 12d then changed the exactness contract itself instead of the atoms: the legacy token-match path stays available, but a new strict budgeted path accepts an atom only if both block-level and activation-conditioned output relMSE stay within hard budgets and otherwise falls back first to a small residual phrase bank before literals. This does show that the contract change is safe: on the first honest `16`-window gate, `budgeted_exact` moves only from `PPL 2.8055` to `2.8069` at essentially unchanged throughput and peak VRAM. But the economics barely move at all: average program length remains `12.9983 / 12.9985`, atom calls are effectively zero, residual calls only `0.0017 / 0.0014`, and accepted-budget token coverage stays around `4e-4`. WAL-SBC Step 12e then tested the exact opposite side of that question with an empirical CDF profiler plus a narrow targeted tune: percentile-derived budgets immediately activate the non-literal path (`mean_nonliteral_calls ~ 3.85`, `avg_program_length ~ 4.8` on the best finalists), proving the contract can indeed buy short programs. But those same raw percentile budgets fail the sacred quality bar badly (`16`-window `PPL delta vs strict legacy ~ +0.37`). That cleanly justifies WAL-CDA as the next move: the remaining problem is no longer how to loosen the contract, but how to replace the atom basis with an economically constrained context-aware layer that can buy those economics without the quality collapse. Concretely, that CDA layer must not treat discrete IDs as numeric coordinates or consume the full live layer input. It should combine embeddings for discrete identities with a cheap detached activation summary, then apply only a clipped factorized low-rank delta over a shared base prior. Its usefulness signal must be a soft expected-cost surrogate under the same budget contract and must compare against the best old accepted path on the existing budgeted basis, not only against strict legacy.
- **M27**: WAL grammar induction. Train a small transformer on `(stage, id)` sequences to obtain a learned token embedding and a per-layer grammaticality proxy.
- **M28**: component-specific dialects (`WAL-Attn`, `WAL-MLP`, `WAL-Head`, `WAL-Norm`).
- **M29**: WAL-to-WAL translation between two related models as a weight-merging substitute.
- **M30**: production-ready inference path and first public release.
