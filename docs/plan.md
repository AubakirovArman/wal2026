# Plan — DRL v2 milestones

> Modular note: when roadmap details stop fitting comfortably here, split them into `docs/roadmap/` and keep this file as the master checklist.

## M0 — Scaffold (2026-04-17)

- [x] Create project folder, README, docs.
- [x] Freeze design decisions in `decisions.md`.
- [ ] Pick calibration dataset (WikiText-2 train slice, ~2k sequences × 2048 tokens, matching v1).
- [x] Confirm Llama 3.3 70B Instruct weights available locally and loadable with streaming / multi-GPU HF runtime.

## M1 — Variable-depth encoder on a toy layer

- [x] Implement `src/route_encoder.py` — greedy residual encoder with `L_max=12`, per-family ladder, stop threshold, `stop_depth` output.
- [x] Unit test on a single synthetic matrix: round-trip relMSE < 0.01 %.
- [x] Port Route B's guarded row-scale to v2 as-is.
- [x] End-to-end encode one `mlp_up` layer of Llama 3.3 70B, measure relMSE distribution.
- [x] Extend the check to all 560 block linears; worst relMSE stays below `4.3e-5`.

## M2 — ID codebook

- [x] Implement `src/codebook.py` — collect unique routes, assign IDs, emit codebook table.
- [x] Measure `M_f` per family on a 3-layer slice first, then on the full 70B.
- [ ] Decide global vs per-family IDs based on `M_f` spread.
- [ ] Implement `src/pack.py` for the final on-disk format (`ids.pt`, `codebook.pt`, `ladders.pt`, `manifest.json`).

## M3 — Reference runtime

- [x] `PackedIDRouteLinear` — gather → reconstruct → matmul in plain PyTorch + CUDA.
- [x] `replace_layers()` — swap targeted block `nn.Linear` modules to packed runtime layers.
- [ ] Sanity generation on 3 prompts — output must be coherent.
- [x] Short PPL gate on WikiText-2 — not catastrophic; 16-window raw gate shows gap `-0.0001`.

## M4 — Full validation

- [x] Full WikiText-2 PPL over the entire cached raw test set — dense `3.4304`, routed `3.4350`, gap `0.0046`.
- [x] HumanEval pass@1 via HF `lm_eval` — full `164`-task run: dense `0.7317`, routed `0.7195`, gap `-0.0122`; elapsed `1337.0 s` dense / `1169.0 s` routed.
- [x] Throughput sweep (bs ∈ {1, 4, 8}, ctx ∈ {512, 2048, 8192}) on the reference runtime.

## M5 — Fused Triton kernel (if M4 passes quality gate)

- [ ] `FusedIDRouteLinear` — fused gather + reconstruction + matmul.
- [ ] Re-measure throughput; require ≥ 0.8× dense on a single H200, including `bs=1` latency.

## M6 — Current Runtime Frontier (2026-04-19)

Historical note: M0--M5 above capture the original DRL-v2 launch plan. The active frontier has moved. The current runtime work is split into an operational branch and a packed-research branch.

- [x] M20 fast reconstruct: establish `full_weight_fast` / `per_group_fast` as the fair packed baseline. Microbench on `8192 x 8192` cut `full_weight` `33.74 ms -> 24.95 ms` and `per_group` `35.03 ms -> 26.56 ms` at unchanged relMSE.
- [x] M21/M22/M24 stage-control sweep: prove that residual-stage count is a real but blunt latency lever. Conservative per-layer `cos >= 0.999` is the only current "almost no quality loss" frontier (`2.9614` PPL / `391.47 tok/s` vs `2.9487` / `385.47` baseline on `first8_qk_gu`).
- [x] M23 influence grammar: confirm that hotness lives at the stage-local `(stage, id)` level rather than in a merged layer vocabulary.
- [x] M25 persisted encoding infrastructure: make same-encoding runtime comparison the default methodology for packed experiments.
- [x] Exact packed hot/cold win: same-encoding `full_weight_hot` beats `full_weight_fast` on validated slices, including `l54_q_gu` `16` windows (`933.41 -> 962.71 tok/s`) at unchanged `PPL = 2.7996`.
- [x] Triton same-encoding check: calibrated `triton_block_rvq` is runnable now, but still clearly slower than the packed fast path on the same encodings.
- [x] Compare harness fix: restore the real operational ceiling. Current full-model baseline-quality frontier is `eager-bf16` (`2.7775` PPL / `1240.29 tok/s`), while `eager-hybrid` is the speed/VRAM frontier with quality cost (`2.8527` / `1402.63 tok/s`).
- [ ] Implement a fused hot/cold packed kernel or cached-hot-block path on top of persisted encodings.
- [ ] Re-run the same-encoding `l54_q_gu` `16`-window gate and push the packed exact path beyond `1000 tok/s` at unchanged perplexity.
- [ ] Extend same-encoding packed comparisons to more attention and MLP slices before promoting any packed runtime as a general frontier.

## M26 — Fused exact hot/cold runtime (in progress, 2026-04-19)

Framing: this is the first compiler optimisation in the WAL toolchain. It must be exact against `full_weight_fast` on the same persisted encodings, not approximate.

- [x] M26 stage 1: allocation-free hot/cold reconstruct (`full_weight_hot_v2`). Reuse a single cached recon buffer per group; initialise via `index_copy_`, accumulate further stages via `index_add_`. Eliminates ~K-1 per-stage `torch.empty((stage_size, block_size))` allocations (about `470 MB` each for `gate_proj/up_proj`).
- [x] M26 stage 1 validation (partial): on `l54_q_gu` 16w, exact-equal vs `full_weight_hot` (`PPL = 2.7996`, layer rel-MSE `4.193e-02`); throughput `971.28` tok/s (`+0.96%` vs `hot`, `+2.97%` vs `fast`). Stage-1 target `>= 980 tok/s` **not reached** on this single-`q_proj` slice; expected to need stage 2 Triton kernel and/or larger MLP slices.
- [x] M26 stage 2 probe A (rejected): `triton_hot_cold_persistent` = exact tile-local persistent palette + Triton hotprefix kernel. Synthetic one-layer forwards worked, but the honest `l54_q_gu` 16w gate failed (`PPL = NaN`, `188.77 tok/s`, peak `47567.1 MB`), and the later `fast_norm` patch still hit CUDA gather out-of-bounds under the full harness. Treat as DEAD.
- [x] M26 stage 2 probe B0 (scaffold only): `stage_local_hot_cold` kernel over native `stage_ids + full_codebooks + hot_lut + hot_codebooks`. One-hot parity on `l54.gate/up` passed (`4.88e-4 / 2.44e-4` max abs diff), but realistic `m = 2048` microbench is slower than `full_weight_hot_v2` (`58.81 -> 89.96 ms` on `gate_proj`, `58.84 -> 89.77 ms` on `up_proj`), and the honest `l54_q_gu` 16w gate failed (`PPL = NaN`, `258.23 tok/s`, peak `46575.8 MB`). Keep as correctness scaffold, not frontier.
- [x] M26 stage 2 probe B1 (failed narrow gate): `stage_local_hot_cold_b1` = native stage-local hot-palette kernel over `stage_ids + full_codebooks + hot_ids + hot_codebooks`. One-hot parity on `l54.gate/up` remained good (`4.88e-4 / 2.44e-4` max abs diff, `onehot_allclose = True`), and `has_nan = False`, but the strict narrow numeric gate still misses (`max_abs_diff = 0.0625 / 0.03125` on realistic random input), while the realistic `m = 2048` microbench collapses versus `full_weight_hot_v2` (`58.81 -> 569.43 ms` on `gate_proj`, `58.79 -> 569.38 ms` on `up_proj`). Per gating policy, do not run full `l54_q_gu` 16w on this variant.
- [x] M26 stage 2 probe B2 (failed narrow gate): `stage_local_hot_cold_b2` now stages each stage hot palette once per tile and removes repeated hot-vector global loads from the inner loop. On the authoritative `l54.gate/up`, `m = 2048` gate it keeps one-hot parity within `1.953e-3`, stays finite (`has_nan = False`), and reaches `78.84 -> 68.36 ms` on both `gate_proj` and `up_proj` (`1.153x` vs `full_weight_hot_v2`) when the old stage-share skip is disabled. This is still a reject: the per-stage hot-hit rate is only `mean/min = 0.077 / 0.075` on `gate_proj` and `0.077 / 0.074` on `up_proj`, far below the required `>= 0.72`, so full same-encoding `l54_q_gu` 16w was not launched.
- [x] M26 stage 2 probe B3 (failed narrow gate): selection-only B3 = `stage_local_hot_cold_b3`, i.e. B2 kernel plus thresholded stage-local hot selection policy. Runtime now supports non-power-of-2 `hot_topk` by padding staged hot palettes to the next power of two, so the requested `topk = 48` path was checked honestly. Results are strictly negative in the requested `48-64` band: at `m = 2048`, `topk = 48` gives `hit_mean = 0.174 / 0.170`, `hit_min = 0.040 / 0.024`, and `speedup = 0.224x`; `topk = 64` gives `hit_mean = 0.222 / 0.219`, `hit_min = 0.040 / 0.024`, and the same `0.224x` on `gate_proj / up_proj`. No NaNs and one-hot parity remain good (`0.001953125`), but the gate fails decisively.
- [x] M26 B3 upper-bound diagnostic: even pure stage-local `count` selection does not rescue this idea at small `k`. The empirical upper bound is only `hit_mean ~0.312` at `topk = 64`; pushing to `topk = 128` raises coverage only to `~0.578` while speed collapses to `0.133x`; `topk = 256` reaches full coverage but only `0.128x`. Therefore the target `hot_hit_rate >= 0.70-0.75` is unattainable for `k <= 64` on these layers under the current encoding.
- [x] M27 RRF step 1a (failed offline viability gate): `Route Register File + spilling` on `l54.gate/up` was tested offline first, using persisted activation-weighted influence, occurrence counts, and structural per-stage interference at `tile_size = 256` (the `block_n` of the B2 kernel). Mean hit rate over 12 stages is only `0.293` for RRF at `cap = 64` (vs `0.311` for plain `topk_count`) and `0.560` at `cap = 128` (vs `0.577` for `topk_count`). The key blocker is structural, not algorithmic: `avg_id_tile_occupancy = 1.000` on every stage, so the interference graph is effectively complete and the allocator degenerates to `topk_by_influence`, which is itself worse than count on these MLP layers. By policy, do not proceed to an RRF runtime/kernel implementation on the current encoding.
- [x] M27 PTDP step 2a (failed offline viability gate): `per_tile_dynamic_palette` was tested directly on `l54.gate/up` with tile shape `64 x 256`, selecting tile-local `topk = 32/48/64` ids by activation-weighted influence. The result is still decisively negative. Mean tile count-hit across stages is only `0.286 / 0.393 / 0.488` on `gate_proj` and `0.289 / 0.394 / 0.490` on `up_proj`; the best stage mean seen anywhere is only `0.494`, and the fraction of tiles with `count_hit >= 0.75` is `0.000` for every checked `topk`. Therefore PTDP Step 2b (runtime integration on top of the B2 kernel) was not launched.
- [x] M27 FGRL step 3a (mixed diagnostic, fails sparsity/runtime gate): `Finer-Grained Route Language` re-encoded `l54.gate/up` from the current total `12 x 256` config to a total `20 x 80` config (`stages_per_split = (5,5,5,5)` at `product_splits = 4`). This improves quality metrics on the stable `4`-window `full_weight_fast` gate: `avg_rel_mse 0.0390 -> 0.0172`, `PPL 2.4054 -> 2.3928`, and tile `top64` share `0.495 -> 0.909`. But it fails the actual sparsity objective: normalized tile occupancy gets worse, not better (`0.853 -> 0.996`), because almost the entire new 80-id alphabet is live in every tile; throughput also regresses (`1483.29 -> 1309.01 tok/s`), bits/weight rise (`3.004 -> 5.003`), and eval peak VRAM rises slightly (`46486.8 -> 46598.5 MB`). Therefore FGRL Step 3b runtime integration was not launched.
- [x] M27 WAL-SS step 4a (lossless macro syntax gate): `m27_wal_ss_proto.py` mined up to `128` frequent subsequences of length `3-5` over the current `12 x 256` stage-id programs on `l54.gate/up` and re-expressed the layers as `macro calls + literal args`. Quality stayed exact by construction (`exact_stage_ids = True`, `recon_rel_mse = 0.0`, and identical `PPL = 2.4081`, `tok/s = 1477.69`, peak `46486.8 MB` on the stable `4`-window gate), but the syntax layer is structurally ineffective: average program length only drops from `12` to `11.9985 / 11.9987`, compression ratio stays `0.999875 / 0.999889`, and macro token coverage is only `0.000176 / 0.000158`. The underlying reason is that the current cache already has almost no repeated full programs (`7,340,032` blocks per layer with `7,340,018 / 7,340,015` unique programs). Therefore WAL-SS in this exact macro-only form is not yet a new ISA baseline.
- [x] M27 WAL-HP step 5a (cross-layer shared-subroutine gate): `m27_wal_hp_proto.py` mined `256` shared subsequences of length `3-6` across `l53/l54 gate/up`, then rewrote each block program as `CALL shared_subroutine` or `LITERAL` and checked exact decode plus the stable `4`-window `full_weight_fast` gate. The wrapper stays exact on all four encoded layers (`exact_stage_ids = True`, `recon_rel_mse = 0.0`, and identical `PPL = 2.4258`, `1216.91 tok/s`, peak `47530.4 MB` vs current packed), and `136 / 256` selected subroutines are truly shared across at least two targets. But the structural gain is still trace-level: global `call_coverage` is only `0.000166`, and per-layer average high-level program length stays at `11.9973-11.9992` out of raw `12`. Therefore WAL-HP Step 5b was not launched.
- [x] M27 WAL-LRT step 6a (learned-template gate): `m27_wal_lrt_proto.py` trained `256` categorical full-program templates on the current `l54.gate/up` stage-id corpus, then measured both exact `TEMPLATE + corrections` syntax compression and a separate `template_only` approximate encoding on the stable `4`-window `full_weight_fast` gate. Exact WAL-LRT stays identical to current by construction (`PPL = 2.4081`, `1479.74 tok/s`, peak `46486.8 MB`), and it does create more structure than WAL-SS/WAL-HP: average program length drops to `11.7495 / 11.7494` out of `12`, with template token coverage `0.04137 / 0.04130`. But the templates themselves are still too weak semantically: `template_only` match is only about `0.104`, layer relMSE rises to about `1.81`, and `template_only` PPL degrades sharply to `2.9375`. Therefore WAL-LRT Step 6b was not launched.
- [x] M27 WAL-FG step 7a (formal-grammar gate): `m27_wal_fg_proto.py` imposed a shallow `15`-rule grammar over the current `12 x 256` stage-id programs on `l54.gate/up`, parsing each block program into a depth-`4` tree with `7` nonterminal nodes. The exact `RULE + literal corrections` wrapper creates a real formal parse surface, but almost no usable structure: average high-level program length stays at `11.9984 / 11.9988` out of raw `12`, with rule token coverage only `0.000272 / 0.000203`. The separate `grammar_only` approximation remains far outside the quality bar (`recon_rel_mse = 1.2928 / 1.3448`, eval-layer relMSE `~1.31`, `PPL 2.4081 -> 2.6866`) at essentially unchanged throughput and peak VRAM. Therefore WAL-FG Step 7b was not launched.
- [x] M27 WAL-TS step 8a (typed-grammar gate): `m27_wal_ts_proto.py` added a six-type taxonomy (`MLP_GATE/UP x COMMON/MIXED/OUTLIER`) over the current `l54.gate/up` stage-id programs, assigned each block to a type by simple surprisal heuristics, and learned type-conditioned phrase banks. All three local types are active in each layer, and common/outlier blocks separate slightly, but the exact typed wrapper is still almost entirely literal: average high-level program length stays at `11.99815 / 11.99848` out of raw `12`, with typed rule coverage only `0.000308 / 0.000253`. The separate `typed_only` approximation is still far outside the quality bar and is actually worse than WAL-FG on this slice (`recon_rel_mse = 1.6587 / 1.9014`, eval-layer relMSE `~1.75`, `PPL 2.4081 -> 2.7994`) at essentially unchanged throughput and peak VRAM. Therefore WAL-TS Step 8b was not launched.
- [x] M27 WAL-ASM step 9a (classic-assembly gate): `m27_wal_asm_proto.py` rewrote `l54.gate/up` block programs into a classic `LABEL/CALL/JMP/MACRO/RET` surface using `64` learned subroutines and `32` macros. Exact WAL-ASM stays identical to current by construction (`PPL = 2.4081`, `1481.11 tok/s`, peak `46486.8 MB`), but the assembly structure is still weak: average program length is only `11.9288 / 11.9309` out of raw `12`, average calls are `0.0702 / 0.0681`, average jumps only `0.000177 / 0.000174`, call-token coverage only `0.0118 / 0.0114`, and macro body coverage only `0.0717 / 0.0751`. The separate `asm_only` approximation stays far outside the quality bar (`recon_rel_mse = 1.8789 / 1.8749`, eval-layer relMSE `~1.85`, `PPL 2.4081 -> 2.9601`). Therefore WAL-ASM Step 9b was not launched.
- [x] M27 WAL-LDI step 10a (learned-discrete-ISA gate): `m27_wal_ldi_proto.py` introduced the first explicitly designed two-level ISA on `l54.gate/up`: `4` learned semantic families per layer plus `4 x 4 x 4 = 64` family-conditioned low-level atoms, for `68` instructions per layer total. This is the first post-hoc probe that produces a real semantic partition rather than only a wrapper: all four families are active (`family_entropy ~1.14`), and a small `COMMON_CORE` cluster separates strongly (`margin ~6.36 / 6.11`). But the exact surface still fails structurally: average program length rises to `12.9969 / 12.9971` out of raw `12` because every block pays a family header while low-level atoms fire only `0.00312 / 0.00289` times per block, with token coverage only `0.000521 / 0.000483`. The separate `ldi_only` approximation improves over WAL-ASM but is still outside the quality bar (`recon_rel_mse = 1.7697 / 1.8243`, eval-layer relMSE `~1.77`, `PPL 2.4081 -> 2.8318`). Therefore WAL-LDI Step 10b was not launched.
- [x] M27 WAL-E2E step 11a (end-to-end semantic-encoder gate): `m27_wal_e2e_proto.py` moved the semantic-ISA idea into the encoder loop itself. It trains a small semantic encoder plus family-conditioned atom bank jointly on sampled block programs with token reconstruction, semantic neighborhood consistency, family cohesion, and anti-collapse regularization. This does keep all four families alive in the authoritative run (`active_family_count = 4` on both layers, entropy `1.147 / 0.802`), so the semantic partition is no longer only post-hoc. But it still does not yield a usable ISA. Exact program length rises further to `12.9988 / 12.9988` out of raw `12`, low-level atoms fire only `0.00119 / 0.00118` times per block, and token coverage drops to `0.000199 / 0.000197`, which is even worse than WAL-LDI structurally. The separate `e2e_only` approximation also remains outside the quality bar (`eval-layer relMSE ~1.70`, `PPL 2.4081 -> 2.8847`), so the first end-to-end objective is another honest negative baseline rather than a breakthrough. Therefore WAL-E2E Step 11b was not launched.
- [x] M27 WAL-DR step 12a (direct-block-reconstruction gate): `m27_wal_dr_proto.py` kept the same semantic-encoder / family-conditioned atom surface as WAL-E2E, but changed the controlling objective itself: direct final-block reconstruction under the current Block-RVQ surface plus an explicit program-cost proxy, with semantic regularizers demoted to secondary terms. This change matters. All four families stay active (`family_entropy 1.119 / 1.249`), and the approximate `dr_only` surface is materially better than both WAL-LDI and WAL-E2E (`eval-layer relMSE ~1.54`, `PPL 2.4081 -> 2.7325`). But the exact ISA is still not structurally usable: average program length remains `12.9984 / 12.9988` out of raw `12`, low-level atoms fire only `0.00156 / 0.00117` times per block, and low-level token coverage is still only `0.000260 / 0.000195`. Therefore WAL-DR Step 12b was not launched.
- [x] M27 WAL-LO step 12b (layer-output-reconstruction gate): `m27_wal_lo_proto.py` moved the main loss one step closer to downstream behavior by reconstructing activation-conditioned layer-output contributions on real captured layer inputs, while adding stronger exact-program cost terms and an explicit atom-vs-literal decision head. The result is another honest negative frontier. Exact quality stays identical by construction, but the approximate `lo_only` surface is slightly worse than WAL-DR (`PPL 2.4081 -> 2.7463`), and the exact program still collapses to almost-all-literal fallback: average program length is `13.0018 / 13.0000` out of raw `12`, low-level calls are only `0.0140 / 0.0000`, and low-level token coverage only `0.00102 / 0.00000`. Therefore simply moving the loss to layer-output behavior is still not enough to create a short exact ISA.
- [x] M27 WAL-LHA step 12c (learnable-high-expressivity-atoms gate): `m27_wal_lha_proto.py` kept the WAL-LO objective surface, but replaced weak static atoms with expressive context-dependent atom modules that generate token logits on top of a learned base phrase prior, plus a separate atom-selection loss and a direct atom-usage bonus. This gives the best approximate surface in the current objective line (`PPL 2.4081 -> 2.7179`, slightly better than WAL-DR and better than WAL-LO), and sample layer-output relMSE improves materially (`1.3418 / 1.2052`). But the exact program is still not economically useful: average program length remains `13.0006 / 13.0000` out of raw `12`, low-level calls are only `0.0083 / 0.0000`, and exact fallback remains overwhelmingly literal. Therefore stronger atom expressivity alone is still not enough to create a short exact ISA.
- [x] M27 WAL-SBC step 12d (strict-budgeted-exactness gate): `m27_wal_sbc_proto.py` kept the WAL-LHA expressive atoms but changed the contract itself. Alongside the old strict legacy path, it introduced a `budgeted_exact` path that accepts an atom only if both block-level and activation-conditioned output relMSE stay within strict budgets, otherwise falls back first to a small residual phrase bank and only then to literals. On the first honest `16`-window gate this contract is safe but still too conservative: `strict_legacy` gives `PPL = 2.8055`, `1001.80 tok/s`, peak `69447.3 MB`, while `budgeted_exact` gives `PPL = 2.8069`, `1001.63 tok/s`, the same peak memory, and average program length still `12.9983 / 12.9985`. Atom calls remain effectively zero and residual calls only `0.00172 / 0.00144`, so revising exactness is the right frontier question, but the first strict budget geometry does not yet buy economically useful atom usage.
- [ ] M26/M27 next step: stop treating smaller alphabets, exact macro wrappers, exact shared-subroutine layers, weak learned-template banks, shallow fixed grammars, weak heuristic type systems, explicit classic-assembly wrappers, or post-hoc learned semantic partitions by themselves as evidence of hardware fit. The controlling question is now whether any representation can create either (a) lower normalized tile occupancy or (b) genuinely reusable higher-order programs with non-vanishing coverage and non-catastrophic semantics, instead of almost-fully-live alphabets, almost-fully-unique stage programs, mostly-literal parse trees, type labels that do not materially carry behavior, CALL/JMP graphs that exist only at trace level, and semantic families that appear only after the fact but still do not compress the actual program.
- [x] M27 WAL-SBC step 12e (empirical-budget profiler + targeted tune): `m27_wal_sbc_budget_profile.py` first measured the real output-error CDFs of expressive atoms and best residual fallback without retraining, then `m27_wal_sbc_tune_proto.py` ran only the percentile-derived grid on a `4`-window screen and promoted just the best two configs to the honest `16`-window gate. This cleanly falsified the idea that the WAL-SBC deadness was only due to over-tight thresholds: percentile budgets immediately activate non-literal paths (`mean_nonliteral_calls ~ 3.85`, `avg_program_length ~ 4.78 / 4.90` for the top two configs), but sacred quality collapses (`16`-window `PPL delta vs strict legacy = +0.368 / +0.380`). So raw empirical percentile budgets are too loose to serve as the exactness contract.
- [ ] M27 WAL-CDA step 12f: `experiments/m27_wal_cda_proto.py` has now landed as a constrained code scaffold: fixed old LHA/SBC basis, embeddings for discrete IDs, cheap detached activation summary, factorized low-rank clipped delta, soft usefulness surrogate against the best old accepted path, and preserved three-way comparison (`strict_legacy`, old `budgeted_exact`, `budgeted_exact + CDA`). This item stays open until the real `4w`/`16w` gate is run and the same success bar is checked numerically (`avg_program_length < 12`, clearly live non-literal path, near-legacy quality, no layer-quality collapse).
- [ ] M26 broadening: extend same-encoding harness to a multi-layer slice (`first8_qk_gu`) so the packed conclusion is not tied to a single attention/MLP layer.
- [ ] M26 hot-cache VRAM regression: rebuild `_build_hot_cache` so that `full_weight_hot/hot_v2` does not double peak VRAM versus `full_weight_fast` on multi-linear slices (observed `88.9` GB vs `47.2` GB on `l0_qkv_gu`).

## M27 — WAL grammar induction

- [ ] Build a `(stage, id)` token corpus from persisted encodings of all 80 Llama-3.3-70B layers.
- [ ] Train a small (4-6 layer) transformer to predict next-token within a stage; use embeddings as a grammaticality proxy.
- [ ] Add per-layer grammaticality and influence-entropy reports to a new diagnostic experiment.

## M28 — Component-specific WAL dialects

- [ ] `WAL-Attn`: RoPE-aware influence weighting for `q/k`, separate hot/cold policy for `o_proj`.
- [ ] `WAL-MLP`: SwiGLU-aware influence for `gate/up`; explicit residual branch for `down_proj`.
- [ ] `WAL-Head`: vocabulary-aware codebook for `lm_head`.
- [ ] `WAL-Norm`: ternary or 8-bit scale+shift for RMSNorm scales.

## M29 — WAL <-> WAL translation

- [ ] Pick a closely related model pair (e.g. Llama 3.3 70B vs Llama 3.1 70B); persist encodings for both with the same Block-RVQ config.
- [ ] Train a small token-to-token translator on aligned layers.
- [ ] Evaluate as a weight-merging substitute on shared eval gates.

## M30 — Production-ready inference + public release

- [ ] Plug the WAL runtime into a serving stack (vLLM-style or minimal own); demonstrate persistent hot caches and overlapped layer prefetch.
- [ ] Publish first paper version + open release.

## Abort conditions

- If M4 full WikiText-2 PPL gap > 0.3 nats even after per-family ladders up to `L_max=12` + row-scale, stop and revisit the ladder fitter (GPTQ-style error propagation or activation-aware selection from v1 Open Items).
- If `M_f` per family explodes beyond 2^24, stop and add route-merging / pruning before ID assignment.

## Non-goals (explicit)

- Multimodal support.
- Hybrid per-layer selective routes (v1 already covered that path).
- BitNet-style QAT from scratch.
- Production packaging for arXiv / HF release — those happen only after M4 passes.


---

## M54 — WAL-0 Codebook Layer: from codec to language

> Status: in progress. M46-M53 proved WAL-0 scalar quality. Now we build the language layer.

**Goal**: Turn WAL-0 from a raw codec (2 int16 per weight) into a compressed language with a vocabulary of unique programs, exactly as DRL v2 did with routes→IDs.

### M54a — GPU-native unique program mining
- [ ] Implement GPU-native `build_program_codebook(weights, atoms, lmax)` that returns `(unique_programs, program_ids, frequencies)` without CPU copy.
- [ ] Target: run on a full 70B layer (e.g. layer 40 o_proj, ~16M weights) entirely on GPU.
- [ ] Report: vocabulary size, entropy, top-10 program frequencies, coverage (top-K % of weights).

### M54b — Program packing and ID assignment
- [ ] Assign compact IDs (Huffman or ANS or simple uint8/uint16) based on frequency.
- [ ] Implement `pack_programs(program_ids)` → bit-packed tensor.
- [ ] Measure effective bits-per-weight after codebook + ID packing.

### M54c — Decode from codebook
- [ ] Triton kernel that decodes directly from `(program_id, codebook)` without materializing full programs first.
- [ ] Validate: same reconstruction as raw WAL-0 decode.

## M55 — WAL-0 Variable Length: stop_depth economics

> Status: pending M54.

**Goal**: Not every weight needs lmax=2. Some need 0, 1, or 2 steps. Variable length saves bits and creates structure.

### M55a — GPU-native early stopping encode
- [ ] Modify `wal_encode_scalar_fused` to accept a `threshold` parameter.
- [ ] If residual² < threshold² after step k, stop. Return `actual_length ≤ lmax`.
- [ ] Measure: distribution of stop_depth across layers (% at depth 0, 1, 2).

### M55b — Variable-length decode kernel
- [ ] Triton kernel that reads `program_length` per weight and loops only that many times.
- [ ] Compare speed vs fixed-lmax kernel.

### M55c — Full 70B PPL with variable length
- [ ] Encode full model with early stopping.
- [ ] Run WikiText-2 PPL. Target: gap ≤ +0.1% vs fixed lmax=2.

## M56 — WAL Grammar: structure in the program stream

> Status: pending M55.

**Goal**: Analyze the program stream as a language. Find reusable patterns, n-grams, subsequences.

### M56a — Program corpus analysis
- [ ] Build per-layer `(atom_id, sign)` token corpus.
- [ ] Measure: unigram/bigram entropy, repeat rate, spatial correlation.
- [ ] Compare to DRL v2 route corpus (from M23).

### M56b — Shared subroutines (WAL-HP v2)
- [ ] Mine frequent subsequences of programs across layers.
- [ ] Build CALL/LITERAL surface. Target: >5% token coverage (vs 0.01% in first WAL-HP).

### M56c — Learned templates (WAL-LRT v2)
- [ ] Train categorical templates over program corpus.
- [ ] Target: template-only coverage >20% with PPL gap < +0.5 nats.

## M57 — WAL Component Dialects

> Status: pending M56.

- [ ] `WAL-Attn`: separate atom tables for q/k/v/o with RoPE-aware calibration.
- [ ] `WAL-MLP`: SwiGLU-aware atoms for gate/up, residual for down.
- [ ] `WAL-Head`: vocabulary-aware codebook for lm_head.

## M58 — Production inference runtime

> Status: pending M57.

- [ ] Fused decode+matmul kernel (not just decode).
- [ ] Persistent hot caches for frequent programs.
- [ ] Single-H200 throughput target: ≥ 0.8× dense.

---

## WAL-0 Abort Conditions

- If M54 codebook vocabulary explodes beyond 2^16 (65K unique programs per layer) even with K=128, lmax=2 → the program space is too diffuse. Need smaller K or structured atoms.
- If M55 variable-length average depth stays > 1.9 (i.e. almost all weights use 2 steps) → early stopping doesn't buy compression. Need better atoms or higher lmax.
- If M56 grammar analysis shows entropy near max (no structure) → WAL-0 programs are too unique to form a language. Need to change ISA (vector atoms, hierarchical, etc.).



---

## M54-M56 — WAL-0 Language Layer (completed 2026-04-20)

### M54a: Codebook Mining ✅
- Layer 40 o_proj (67M weights): **1,079 unique programs** (0.0016% vocabulary ratio).
- Entropy: 9.13 bits. Top-1024 coverage: 99.98%.
- **Result**: WAL-0 has a tiny vocabulary. Codebook layer is viable.

### M54b: Codebook Decode ✅
- Three decode strategies tested:
  1. Atom-lookup via codebook: **23,945 Mw/s**
  2. Precomputed recon lookup: **1,144,207 Mw/s** (~1.1 TW/s!)
  3. Triton raw programs: 93.5 Mw/s
- All exact (max error 0.00).
- **Result**: Precomputed recon lookup is revolutionary. Decode = single `index_select`.

### M55a: Variable Length ✅
- Early stopping with thresholds tested.
- threshold=0.005: 96% weights stop at depth 1, save 0.24 bits (2.3%).
- threshold=0.010: 96.1% stop at depth 1, save 0.38 bits (3.8%).
- **Trade-off**: small threshold (<0.002) → no compression gain due to overhead. Large threshold → quality collapse (relMAE 0.177 at threshold=0.05).
- **Result**: Variable length works but needs careful threshold tuning or adaptive per-layer/per-row thresholds.

### M56a: Grammar Analysis ✅
- **Conditional entropy H(prog[i+1]|prog[i]): 9.288 bits** vs unigram 9.298 bits.
- Predictability gain: **0.010 bits (0.1%)** — essentially zero.
- Spatial autocorrelation: 0.014 (row), 0.006 (col) — negligible.
- Repeat rate: 0.21% — neighbors almost never share the same program.
- Unique bigrams: 80.9% of possible pairs appear.
- **Result**: WAL-0 program stream is **i.i.d.** No grammar, no spatial structure, no n-gram patterns.

### Key Conclusion from M54-M56

**WAL-0 is an excellent codec, not a language.**

- It achieves near-dense PPL (2.7858 vs 2.7805).
- It has a tiny vocabulary (~1000 programs per layer).
- It decodes at 1.1 TW/s via precomputed lookup.
- But its programs do not form a structured language. They are independent, uncorrelated, and unpredictably distributed.

**This mirrors DRL v2's experience**: M23-M27 also found that the stage-id stream was almost maximally unique (WAL-SS coverage 0.000176, WAL-HP call coverage 0.000166). WAL-0 repeats this pattern at the scalar level.

**Implication**: To create a true weight language, the ISA itself must enforce structure. Post-hoc grammar mining on a flat greedy residual encoding does not work.

---

## M57 — Full 70B Codebook Encode + PPL

**Goal**: Prove that codebook-based WAL-0 scales to full model without quality loss.

- [ ] Encode all 540 non-spiky params with per-param k-means + codebook.
- [ ] Apply precomputed recon lookup (not raw atom gather) for speed.
- [ ] Run full WikiText-2 PPL. Target: gap ≤ +0.2% vs dense.
- [ ] Measure total encode time. Target: ≤ 40 min (same as M53c).

## M58 — Creating Structure in WAL

**Goal**: Since post-hoc grammar mining failed (M56a), enforce structure at the ISA level.

### M58a — Constraint-based encoding
- [ ] Encode with explicit structure constraints: e.g., force spatial smoothness, enforce n-gram patterns, or use a learned prior during greedy search.
- [ ] Measure: does structure enforcement degrade quality? By how much?

### M58b — Hierarchical atoms (WAL-1 reboot)
- [ ] Vector atoms with continuous coefficients (4-bit) and lmax=4-8.
- [ ] Or: tile-local atoms (256-dim vectors) with scalar coefficients.
- [ ] Target: relMSE < 0.01 on vector atoms.

### M58c — Context-dependent encoding
- [ ] Per-row or per-block atom tables (instead of global per-param).
- [ ] Or: atom selection conditioned on row statistics.
- [ ] Measure spatial correlation of programs with context-dependent atoms.

## M59 — Cross-layer Global Codebook

**Goal**: Single codebook for entire model.

- [ ] Pool programs from all layers. Mine global unique programs.
- [ ] Target: global vocabulary < 10K programs for full 70B model.
- [ ] Measure PPL with global codebook + per-layer atom tables.

## M60 — Production Runtime

**Goal**: Fastest possible decode.

- [ ] Precomputed recon lookup kernel (not atom gather).
- [ ] Fused decode+matmul.
- [ ] Single-H200 throughput target: ≥ 0.8× dense.


---

## M60–M77 — WAL v1 Completion (2026-04-20)

### M60 — WAL v2 Scalar Prototype ✅
- Single-call programs with continuous coefficients.
- Layer 40 o_proj: relMSE 0.00001056, output relMSE 0.00001657, corr 1.0.
- Files: `src/wal/v2/encoder.py`, `src/wal/v2/isa.py`

### M61 — WAL v2 Full 70B Encode + PPL ✅
- **PPL 2.7781** vs baseline 2.7805 (delta −0.0024).
- Encode time: 1810s (30 min). 1.33× compression.
- 540 encoded params, 183 skipped.
- Files: `experiments/m61_wal_v2_70b_ppl.py`

### M62 — Grammar & Assembler ✅
- Exact round-trip. 1,299 unique programs / 67M weights per layer.
- Files: `src/wal/v2/grammar.py`, `src/wal/v2/asm.py`

### M63 — VM Runtime ✅
- Reference interpreter + Triton kernels.
- Files: `src/wal/v2/vm.py`, `src/wal/v2/triton_kernels.py`

### M64 — Compression Format v2 ✅
- Binary format with uint4-packed coeff_ids, sparse residual bitmap.
- Files: `src/wal/v2/format.py`

### M65–M69 — Compression Prototypes (negative results) ✅
- M65: Tile/vector quantization — single-layer OK, full PPL toxic.
- M66: PQ prototype.
- M67: Two-tier PQ — 8 bits DEGRADE (3.1137), 12 bits PASS (2.7824).
- M68: SVD — relMSE 0.55-0.99, toxic.
- M69: Position-specific varying K — K=256→3.02 DEGRADE.
- **Lesson: 12 bits/weight is the hard floor for 70B Llama.**

### M70–M73 — Full PPL Validation ✅
- M71: Single-layer PPL does NOT predict full PPL (diff up to 24,500×).
- M73: Two-tier full PPL — 12 bits = hard floor.

### M74 — WAL v1 Two-Term Prototype ✅
- Two-term greedy (32 bits) excellent relMSE.
- Subroutine clustering (256 subs, 12 bits) toxic (relMSE 0.04).
- Lesson: diversity of optimal pairs too high for template clustering.

### M75 — WAL v1 Full 70B PPL ✅
- **PPL 2.7809** vs baseline 2.7805 (delta +0.0004 PASS).
- 35,840 L1 atoms created across 560 layers.
- 1866s encode time.
- **WAL v1 design principle:** semantic/interpretability layer on top of WAL v2. Same quality, richer structure.
- Files: `experiments/m75_wal_v1_70b_ppl.py`, `src/wal/v1/isa.py`

### M76 — Binary Format v1 + Round-Trip Test ✅
- **5/5 PASS**: binary round-trip, text round-trip, hierarchical consistency, binary+hierarchical, text→binary→text.
- Files: `src/wal/v1/format.py`, `experiments/m76_wal_v1_roundtrip.py`

### M77 — PyTorch Integration ✅
- `WALParameter` — lazy decode with cache.
- `WALLinear` — decode on-the-fly per forward.
- `WALCachedLinear` — persistent decode cache.
- `replace_linear_with_wal()` — replace all `nn.Linear` in a model.
- **5/5 PASS**: WALParameter, WALLinear forward, WALCachedLinear, replace_linear, device transfer.
- Files: `src/wal/v1/nn.py`, `experiments/m77_pytorch_integration.py`

---

## Updated Abort Conditions (post-M77)

1. **Compression**: 12 bits/weight is empirically the hard floor. Variable-length and learned coeff tables may squeeze marginally, but order-of-magnitude improvements require ISA changes (not just encoding optimizations).
2. **Quality**: Only full-model PPL is valid. Single-layer metrics, relMSE, and sign agreement are all unreliable predictors.
3. **Structure**: Post-hoc grammar mining on flat greedy encodings does not work (M54-M56, M27 WAL-SS through WAL-LDI all confirmed this). Structure must be enforced at ISA level.

## Next Phases (M78+)

| Phase | What | Status |
|-------|------|--------|
| 7 | Debugger & Inspector | ✅ COMPLETE (M78) |
| 8 | Standard Library | ✅ COMPLETE (M79) |
| 9 | Hardware Backends | ✅ COMPLETE (M80) |
| 10 | Meta-learning | ✅ COMPLETE (M81-M82) |
| 11 | Ecosystem (HF Hub, ONNX) | ✅ COMPLETE (M83) |
