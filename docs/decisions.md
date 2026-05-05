# Design decisions log (ADR-lite)

> Modular note: add new ADR-style decisions as separate files under `docs/decisions/`, and use this file as the master index / compatibility bundle.

Each entry: date, decision, rationale, alternatives considered.

---

## 2026-04-17 — 001 — Dynamic depth, `L_max = 12`

**Decision.** Variable-length ternary routes up to depth 12, with per-weight `stop_depth`.

**Rationale.** v1 (fixed depth 5) hit a PPL ceiling on long context specifically because outlier rows could not extend locally. Fixed depth 7 prototype (`1.7242 / 5.61` vs dense `1.7112 / 5.54`) showed extending depth helps, but *global* long depth wastes bits. Dynamic depth addresses both.

**Alternatives.** Fixed 7, fixed 9, two-tier (5+extra for outliers). Rejected: two-tier doubles the runtime branch logic with no free-parameter advantage over honest per-weight stop_depth.

---

## 2026-04-17 — 002 — Target model: Llama 3.3 70B Instruct (text-only)

**Decision.** Use Llama 3.3 70B Instruct as the primary target, not Gemma 4.

**Rationale.** v1 measurements on Gemma 4 31B IT suffered from completions-harness artefacts on AIME/GPQA. Llama 3.3 70B is text-only, has clean WikiText-2 and HumanEval baselines, and v1 already has measured references (F16 PPL 3.8405; ITQ3_S+imatrix PPL 4.4815).

**Alternatives.** Llama 3.1 8B (too small to be a useful quality target), Qwen 3 (not enough internal baselines yet).

---

## 2026-04-17 — 003 — Two-pass encoding: routes then IDs

**Decision.** Encode every weight to a route, then build a codebook of unique routes and store an ID per weight.

**Rationale.** The set of routes that actually occurs in a real trained model is tiny compared to `3^12`. An ID tensor + shared codebook decouples "language design" (ladders + dynamic depth) from "storage" (IDs), which is the v2 thesis.

**Alternatives.** Store routes directly (v1 style). Rejected: loses the quality diagnostic that codebook entropy provides and makes runtime gather harder to fuse.

---

## 2026-04-17 — 004 — Quality over compression

**Decision.** The project's success metric is PPL gap vs dense, not bpw.

**Rationale.** v1 already shipped a compression-first release. v2 exists to answer a different question: if we stop optimising for bits, can the discrete weight language hit dense-level quality at all? bpw remains a reporting metric.

---

## 2026-04-17 — 005 — File size cap 200 lines

**Decision.** Every `.py` file in `src/` stays ≤ 200 lines; split into modules when longer.

**Rationale.** Matches user preference and keeps the kernel surface auditable.

---

## 2026-04-17 — 006 — GPU mask

**Decision.** Default `CUDA_VISIBLE_DEVICES=2,3,5`; never GPU 7.

**Rationale.** Carried over from user memory preferences.

---

## 2026-04-17 — 007 — Row-normalise and pin the top ladder step

**Decision.** Before route encoding, normalise each weight row by its max absolute value, then fit the ladder with a pinned top scale of exactly `1.0` and a geometric seed (`1.0, 0.5, 0.25, ...`) refined for 20 coordinate-descent iterations.

**Rationale.** The first real Llama probes failed not because the ternary encoder was wrong, but because the ladder fit drifted away from the row-normalised top outliers. Pinning the top step preserves exact coverage of the largest weight in every row and collapses real-tensor relMSE from percent-level errors to `1e-5` scale.

**Alternatives.** Quantile row-scale, quantile ladder seeds, clipping outliers, and leaving the top scale free during refinement. Rejected: they either worsened the hardest attention/MLP tensors or added complexity without improving the final PPL gate.

---

## 2026-04-19 — 008 — Do not treat tile-local hotprefix persistent palettes as M26 stage-2 baseline

**Decision.** The `triton_hot_cold_persistent` branch (exact tile-local palette + hotprefix Triton kernel) is an explored dead-end, not the primary M26 stage-2 path.

**Rationale.** The branch looked attractive because it reuses an existing Triton hotprefix kernel and keeps palettes persistent on GPU. However the honest gate failed. On `l54_q_gu` same-encoding `16`-window evaluation, the first full run produced `PPL = NaN` and only `188.77 tok/s` at peak `47567.1 MB`. After switching the plan builder to the `fast` pre-row-scale tensor, small synthetic forwards still worked, but the full harness remained unstable with CUDA gather out-of-bounds / device-side assert. That makes this path non-operational even before speed is considered.

**Alternatives.** Continue tuning tile width, hot size, or autotuning of the tile-local hotprefix kernel. Rejected for now: the failure mode is not a small performance miss but instability under the honest end-to-end harness. The next M26 stage-2 attempt should stay closer to the original Block-RVQ representation: stage-local fused kernel over native stage ids/codebooks, not whole-tile repacking into local palettes.

---

## 2026-04-19 — 009 — Treat the first native stage-local hot/cold Triton kernel as a scaffold, not as M26 stage-2 closure

**Decision.** Keep `stage_local_hot_cold` as an experimental scaffold, but do not count it as the M26 stage-2 implementation.

**Rationale.** This branch stays on the native Block-RVQ representation (`stage_ids + full_codebooks + hot_lut + hot_codebooks`) and therefore fixes the WAL-level mistake of Variant A. The narrow logic gate passed: on one-hot inputs for `l54.gate_proj` / `l54.up_proj`, the new kernel matches `full_weight_hot_v2` within `4.88e-4 / 2.44e-4` max abs diff. However the honest performance and quality gates still fail. At realistic `m = 2048`, `stage_local_hot_cold` is slower than `full_weight_hot_v2` on both MLP layers (`58.81 ms -> 89.96 ms` on `gate_proj`, `58.84 ms -> 89.77 ms` on `up_proj`). On the full `l54_q_gu` same-encoding `16`-window run it yields `PPL = NaN`, only `258.23 tok/s`, and peak `46575.8 MB`.

**Alternatives.** Continue tweaking this exact kernel surface by changing dot precision or cache dtype. Rejected for now: the problem is not a small arithmetic mismatch but that the kernel still does per-row dynamic LUT gathers inside the inner loop, so it behaves like a scaffold rather than a true fused hot-palette kernel. The next step must explicitly stage the hot palette per sub-stage in fast memory (shared/registers) and use this scaffold only as a correctness reference.

---

## 2026-04-19 — 010 — Do not treat the first “real” stage-local hot-palette kernel as an M26 stage-2 frontier

**Decision.** Keep `stage_local_hot_cold_b1` as an explored B1 probe, but do not promote it to the M26 stage-2 baseline and do not spend a full `l54_q_gu` same-encoding run on this exact implementation.

**Rationale.** This B1 branch removes the explicit `hot_lut` indirection and uses native `stage_ids + full_codebooks + hot_ids + hot_codebooks`, so it is the first attempt at a real stage-local hot-palette kernel rather than the B0 scaffold. The narrow logic gate still passes on one-hot inputs for `l54.gate_proj` / `l54.up_proj`: max abs diff remains `4.88e-4 / 2.44e-4`, and `onehot_allclose = True`. Realistic random inputs stay finite (`has_nan = False`), but the strict B1 gate still fails. Random-input max abs diff remains `0.0625 / 0.03125`, above the intended narrow threshold, and realistic `m = 2048` latency regresses catastrophically versus `full_weight_hot_v2` (`58.81 ms -> 569.43 ms` on `gate_proj`, `58.79 ms -> 569.38 ms` on `up_proj`). By the agreed gating policy, this already disqualifies B1 from a full same-encoding `16`-window compare.

**Alternatives.** Keep tuning block sizes or launch the full harness anyway. Rejected for now: the result is not a near miss but a structural signal that the kernel still performs repeated hot-id comparisons and hot-vector loads inside the inner loop. The next stage-2 probe must stage the hot palette once per stage/tile in shared/register memory and avoid reloading the same hot vectors inside the HOT-size loop.

---

## 2026-04-19 — 011 — Do not promote B2 to the M26 stage-2 frontier either

**Decision.** Keep `stage_local_hot_cold_b2` as the landed staged-hot-palette prototype, but do not treat it as the new M26 stage-2 baseline and do not spend a full `l54_q_gu` same-encoding run on it.

**Rationale.** B2 fixes the local compute shape that B1 still got wrong: hot vectors are staged once per stage/tile and reused via a compact hot-slot lookup, rather than reloaded from global memory inside the HOT loop. The honest B2-specific narrow gate on `l54.gate_proj` / `l54.up_proj` at `m = 2048` stays numerically acceptable on the narrow criteria we already use locally: one-hot max abs diff is `0.001953125`, `has_nan = False`, and with the old stage-share skip disabled the kernel improves latency from `78.84` to `68.36` ms (`1.153x`) versus `full_weight_hot_v2`. The probe still fails the agreed keep/drop rule because the stage-local hot-hit economics are wrong: per-stage hot-hit rate is only about `0.077` mean and `0.074-0.075` minimum, far below the intended `>= 0.72`, and `topk = 32` only raises coverage to about `0.15` while making B2 slower again.

**Alternatives.** Treat the `1.153x` microbench win as good enough and launch the full harness anyway. Rejected for now: that would promote a kernel whose staged palette is barely used by the current encoding. The next M26 step should move from inner-loop surgery to hot-cache selection itself: either rebuild `_build_hot_cache` around true stage-local hit rate, or accept that this encoding does not justify a stage-local hot palette on these layers.

---

## 2026-04-19 — 012 — Do not treat selection-only B3 as the M26 stage-2 fix

**Decision.** Keep `stage_local_hot_cold_b3` as an explored policy variant, but do not promote it to the M26 stage-2 frontier and do not spend a full same-encoding `l54_q_gu` run on it.

**Rationale.** B3 tested the user's concrete hypothesis directly: keep the B2 compute shape, but replace the old weak hot-selection rule with a thresholded stage-local policy. The runtime was extended so non-power-of-2 `hot_topk` is launchable via power-of-two padding, which allowed the requested `topk = 48` check. The honest `m = 2048` gate on `l54.gate_proj` / `l54.up_proj` still fails decisively in the intended `48-64` range. At `topk = 48`, hit rate is only `0.174 / 0.170` mean and `0.040 / 0.024` minimum, with speed only `0.224x` vs `full_weight_hot_v2`; at `topk = 64`, hit rate rises only to `0.222 / 0.219` mean, with the same `0.224x` speed. The more important diagnostic is the count-based upper bound: even pure `count` selection only reaches `~0.312` mean coverage at `topk = 64`; `topk = 128` improves to only `~0.578` and is even slower (`0.133x`). That falsifies the working assumption that hot selection alone can close the gap at small `k`.

**Alternatives.** Keep tuning threshold ratios, score modes, or `topk` within the same B3 family. Rejected for now: the upper-bound diagnostic already shows that the target `hot_hit_rate >= 0.70-0.75` is not reachable for `k <= 64` on these layers under the current encoding. The next step must either change the representation or explicitly retire small-k stage-local hot palettes as a viable M26 stage-2 path on this slice.

---

## 2026-04-20 — 013 — Do not proceed from offline RRF analysis to an RRF runtime branch on the current encoding

**Decision.** Keep `Route Register File + spilling` as an explored WAL idea, but stop after the offline allocator analysis and do not build an RRF runtime/kernel branch on the current `l54.gate/up` encoding.

**Rationale.** The explicit Step 1a check already falsified the premise needed for any Step 1b implementation. Using persisted activation-weighted influence, occurrence counts, and structural per-stage interference at `tile_size = 256`, the interference-aware linear-scan allocator reaches only `0.293` mean hit rate at `cap = 64` and `0.560` at `cap = 128`, both worse than or equal to the much cheaper `topk_count` baseline (`0.311` and `0.577`). The reason is structural: every stage has `avg_id_tile_occupancy = 1.000`, so the interference graph is effectively complete and the allocator collapses to `topk_by_influence`. That makes an RRF kernel branch strictly unjustified on this slice before any implementation work begins.

**Alternatives.** Proceed anyway and hope that a persistent register file in runtime still wins. Rejected for now: the offline gate already says there is no useful register-pressure signal to exploit. The next viable direction must change the representation or the compute schedule itself, not just add a more complicated selection layer on top of the same stage vocabulary.

---

## 2026-04-20 — 014 — Do not proceed from PTDP offline analysis to a per-tile dynamic-palette runtime branch on the current encoding

**Decision.** Keep `Per-Tile Dynamic Palette (PTDP)` as an explored WAL idea, but stop after the offline tile-local coverage analysis and do not build a `per_tile_dynamic_palette` runtime/kernel branch on the current `l54.gate/up` encoding.

**Rationale.** PTDP was the most direct attempt to remove the structural weakness exposed by RRF: instead of one stage-global cached vocabulary, each `64 x 256` tile gets its own small palette chosen by tile-local activation-weighted influence. The offline gate still fails decisively. Mean tile count-hit across stages reaches only `0.286 / 0.393 / 0.488` on `gate_proj` and `0.289 / 0.394 / 0.490` on `up_proj` for `topk = 32 / 48 / 64`. The best stage mean anywhere in the run is only `0.494`, and the fraction of tiles with `count_hit >= 0.75` is exactly `0.000` for every checked `topk`. So even after moving from stage-level to tile-level selection, the current encoding keeps mass too diffuse inside the tile for a small dynamic palette to be useful.

**Alternatives.** Proceed anyway and hope that kernel fusion closes the gap. Rejected for now: the offline gate is already below the needed operating regime by a wide margin, so runtime work would only beautify a negative premise. The next viable direction must first create much stronger lexical concentration inside the tile, not just repackage the same diffuse token mass.

---

## 2026-04-20 — 015 — Do not treat FGRL `20 x 80` as a solved runtime path for small cached vocabularies

**Decision.** Keep `Finer-Grained Route Language (FGRL)` as an interesting encoding-quality direction, but do not promote the tested `20 x 80` variant to a runtime branch for `l54.gate/up` and do not treat it as evidence that the small-vocabulary problem is solved.

**Rationale.** The tested FGRL candidate re-encodes the two MLP layers from the current total `12 x 256` configuration to a total `20 x 80` configuration (`stages_per_split = (5,5,5,5)` with `product_splits = 4`). It clearly improves approximation quality on the slice: average layer `rel_mse` drops from `0.0390` to `0.0172`, and the stable `4`-window `full_weight_fast` gate improves from `PPL = 2.4054` to `2.3928`. It also raises tile `top64` share from `0.495` to `0.909`. However the intended mechanism does not materialize. Normalized tile occupancy rises from `0.853` to `0.996`, meaning that almost the entire new 80-id alphabet is still live in each tile. The higher `top64` share comes mostly from shrinking the alphabet until `64` is nearly the whole vocabulary, not from creating true lexical sparsity. At the same time the candidate is more expensive operationally: `bits/weight` rises from `3.004` to `5.003`, throughput drops from `1483.29` to `1309.01 tok/s`, and eval peak VRAM rises slightly (`46486.8 -> 46598.5 MB`).

**Alternatives.** Promote FGRL anyway because the two-layer PPL improves. Rejected for now: this would confuse a quality-improving re-encoding with a hardware-aligned sparse language pass. The next viable direction must reduce normalized occupancy and preserve runtime economics, not just buy a smaller alphabet at significantly higher storage cost.

---

## 2026-04-20 — 016 — Do not treat the first WAL-SS macro layer as a new ISA baseline

**Decision.** Keep `WAL with Structured Syntax (WAL-SS)` as a promising language-design direction, but do not promote the first lossless macro-only prototype to a new baseline ISA or runtime branch.

**Rationale.** The first prototype intentionally tested the weakest, safest syntax layer: keep the current `12 x 256` encoding exactly as it is, mine up to `128` frequent subsequences of length `3-5`, and re-express each block program as `macro calls + literal args` with exact decode back to the original stage-id stream. That exactness gate passes perfectly: stage IDs reconstruct exactly, `recon_rel_mse = 0.0`, and the stable `4`-window `full_weight_fast` gate is identical to the current packed path (`PPL = 2.4081`, `1477.69 tok/s`, peak `46486.8 MB`). The problem is structural utility, not correctness. Each layer contains about `7,340,032` block programs, and almost every full program is unique (`7,340,018 / 7,340,015`). As a result the macro layer barely changes the syntax surface: average program length moves only from `12` to `11.9985 / 11.9987`, compression ratio stays `0.999875 / 0.999889`, and macro token coverage is only `0.000176 / 0.000158`.

**Alternatives.** Treat exact decode plus nonzero macro usage as enough evidence that WAL-SS already works. Rejected for now: many macros are technically used, but only at vanishing frequency, so the language has not become meaningfully more structured. The next viable syntax-level step must create reusable higher-order programs rather than merely proving that a lossless dictionary wrapper can sit on top of an almost fully unique token stream.

---

## 2026-04-20 — 017 — Do not treat the first WAL-HP shared-subroutine layer as a new hierarchical ISA baseline

**Decision.** Keep `WAL with Hierarchical Programs (WAL-HP)` as a promising language-design direction, but do not promote the first exact shared-subroutine prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-HP probe strengthened the WAL-SS premise in the most conservative exact way available: keep the current `12 x 256` encoding exact, extend the target slice to four MLP layers (`l53/l54 gate/up`), mine `256` shared subsequences of length `3-6` across the combined block-program corpus, and re-express each program as `CALL shared_subroutine` or `LITERAL`. Exactness again passes perfectly on all four encoded layers (`exact_stage_ids = True`, `recon_rel_mse = 0.0`), and the stable `4`-window `full_weight_fast` gate is identical to the current packed path on that four-layer slice (`PPL = 2.4258`, `1216.91 tok/s`, peak `47530.4 MB`). There is also real cross-target reuse in the narrow sense: `136 / 256` selected subroutines are used in at least two targets, and the strongest motif (`[128,128,128]`) appears in all four. But the structural effect is still vanishingly small. Global `call_coverage` is only `0.000166`, and average high-level program length stays essentially flat at `11.9973-11.9992` versus raw `12`. So the hierarchy exists only as trace motifs, not as a meaningful new program surface.

**Alternatives.** Treat the existence of cross-layer shared subroutines as enough evidence that hierarchical WAL already works. Rejected for now: the reuse is real but too sparse to change the language economics. The next viable hierarchical step must introduce stronger structure, such as parameterized or approximate subroutines, learned templates, or an encoder that itself creates repeated higher-order blocks.

---

## 2026-04-20 — 018 — Do not treat the first WAL-LRT template bank as a new high-level ISA baseline either

**Decision.** Keep `WAL with Learned Route Templates (WAL-LRT)` as a promising language-design direction, but do not promote the first learned-template prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-LRT probe finally moved beyond mined exact repeats and allowed the language to create its own reusable surface. On `l54.gate/up`, it trained `256` categorical full-program templates over the current `12 x 256` block-program corpus, then evaluated two surfaces separately: an exact `TEMPLATE + literal corrections` wrapper and a `template_only` approximate encoding. This is the first syntax probe that does create visibly more structure than the earlier exact wrappers: average high-level program length falls to `11.7495 / 11.7494` out of raw `12`, and template token coverage rises to `0.04137 / 0.04130`, both far above WAL-SS and WAL-HP. Exact decode also keeps the stable `4`-window `full_weight_fast` gate identical to current (`PPL = 2.4081`, `1479.74 tok/s`, peak `46486.8 MB`). But the learned templates themselves are still semantically weak. The nearest template matches only about `10.4%` of tokens on average (`template_only_avg_hamming ~ 10.75 / 12`), reconstruction relMSE jumps to about `1.84 / 1.84`, and `template_only` PPL degrades sharply from `2.4081` to `2.9375`. So the first learned bank creates a little syntax, but not a meaningful high-level program language that carries model behavior by itself.

**Alternatives.** Treat the modest compression win of exact `TEMPLATE + corrections` as enough evidence that WAL-LRT already works. Rejected for now: the syntax gain is real but still small, and it comes only with a large correction tail; when corrections are removed, quality collapses. The next viable template-level step must make templates themselves more expressive, for example through parameterized templates, weight-space templates, continuous or low-rank corrections, or an encoder that explicitly optimizes for template-friendly programs.

---

## 2026-04-20 — 019 — Do not treat the first WAL-FG formal grammar as a usable ISA baseline either

**Decision.** Keep `WAL as a Formal Grammar (WAL-FG)` as an explored language-design direction, but do not promote the first shallow grammar prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-FG probe tested the next conservative language hypothesis after WAL-LRT: impose an explicit formal grammar on the current `12 x 256` stage-id stream, instead of only mining templates or subsequences from it. On `l54.gate/up`, `m27_wal_fg_proto.py` defines a `15`-rule grammar over four phrase slots of length `3`, so every block program parses as a depth-`4` tree with `7` nonterminal nodes. Two surfaces were evaluated separately: an exact parse wrapper with `RULE calls + literal corrections`, and a `grammar_only` approximate encoding where each slot is replaced by its nearest learned production. The exact parse does produce a real tree, but it barely changes the language economics: average high-level program length remains `11.9984 / 11.9988` out of raw `12`, average rule calls are only `0.00163 / 0.00121`, and rule token coverage is only `0.000272 / 0.000203`. The approximate surface is less catastrophic than WAL-LRT `template_only` on this narrow slice, but still nowhere near acceptable: reconstruction relMSE rises to `1.2928 / 1.3448` (eval-layer relMSE about `1.31`), and PPL degrades from `2.4081` to `2.6866` at essentially unchanged throughput and peak VRAM. So the first explicit grammar adds formal parse structure, not a semantically useful grammar.

**Alternatives.** Treat the existence of a real parse tree and a somewhat milder quality drop than `template_only` as enough evidence that WAL-FG already works. Rejected for now: the tree is mostly a wrapper around literals, the learned productions fire at vanishing rates, and `grammar_only` quality still violates the project's sacred quality bar. The next viable grammar-level step would need productions that carry semantics materially better than a weak fixed phrase bank.

---

## 2026-04-20 — 020 — Do not treat the first WAL-TS type system as a usable typed ISA baseline either

**Decision.** Keep `WAL with Route Types and Type System (WAL-TS)` as an explored language-design direction, but do not promote the first heuristic typed-grammar prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-TS probe tested the next conservative syntax/semantics hypothesis after WAL-FG: maybe explicit route types can give the language a more meaningful high-level surface even when untyped grammars remain flat. On `l54.gate/up`, `m27_wal_ts_proto.py` defined a six-type taxonomy (`MLP_GATE/UP x COMMON/MIXED/OUTLIER`), assigned each block to a type by average stage-id surprisal, and then learned separate phrase banks for each local type. This does create a real typed label space: all three local types are active in each layer with stable shares of about `48.9% / 40.8% / 10.3%`, and the easy/hard extremes do separate slightly (`type_separation_margin` is positive for `COMMON` and `OUTLIER`). But the exact typed grammar still barely changes the language economics: average high-level program length remains `11.99815 / 11.99848` out of raw `12`, typed rule coverage is only `0.000308 / 0.000253`, and the middle `MIXED` band is not cleanly separated at all (its centroid margin is slightly negative on both layers). The approximate surface is also not acceptable and is actually worse than WAL-FG on this slice: reconstruction relMSE rises to `1.6587 / 1.9014` (eval-layer relMSE about `1.75`), and PPL degrades from `2.4081` to `2.7994` at essentially unchanged throughput and peak VRAM. So the first typed layer creates labels, not a semantically useful typed assembly.

**Alternatives.** Treat the presence of explicit types and partial extreme-case separation as enough evidence that WAL-TS already works. Rejected for now: the type system mostly relabels rarity bands, the exact typed parse remains almost entirely literal, and `typed_only` quality still violates the project's sacred quality bar. The next viable typed step would need types that actually constrain or explain behavior, rather than only partitioning the existing flat token stream.

---

## 2026-04-20 — 021 — Do not treat the first WAL-ASM classic assembly surface as a usable ISA baseline either

**Decision.** Keep `WAL as Classic Assembly (WAL-ASM)` as an explored language-design direction, but do not promote the first `LABEL/CALL/JMP/MACRO/RET` prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-ASM probe asked the strongest surface-level syntax question so far: if the flat `12 x 256` stage-id stream does not expose enough natural structure, can an explicitly classical assembly surface create it? On `l54.gate/up`, `m27_wal_asm_proto.py` learns `64` full-program subroutines, turns them into named labels, macroizes their bodies with `32` macros of length `3-5`, and builds a dominant `JMP` table from consecutive `CALL` sites. The exact wrapper remains correct by construction (`PPL = 2.4081`, `1481.11 tok/s`, peak `46486.8 MB`, identical to current packed), but the new control-flow surface is still trace-level. Average program length stays at `11.9288 / 11.9309` out of raw `12`, average calls are only `0.0702 / 0.0681`, average jumps only `0.000177 / 0.000174`, and call-token coverage only `0.0118 / 0.0114`. Macroized subroutine bodies exist, but even there macro-body coverage is only `0.0717 / 0.0751`, with just `8 / 32` macros actually used. The approximate `asm_only` surface is even less acceptable than earlier syntax probes: token match is only `0.08595 / 0.08580`, reconstruction relMSE rises to `1.8789 / 1.8749` (eval-layer relMSE about `1.85`), and PPL degrades from `2.4081` to `2.9601` at essentially unchanged throughput and peak VRAM.

**Alternatives.** Treat the existence of explicit labels, subroutines, macroized bodies, and a non-empty jump table as enough evidence that classical weight assembly already works. Rejected for now: the assembly objects are formal rather than semantic. The first WAL-ASM surface mostly renames weak nearest-template structure, while `CALL/JMP` usage remains too sparse to carry behavior. The next viable assembly-level step would need either a representation that naturally yields much higher control-flow coverage or semantics-bearing constructs that make subroutine choice and jump targets explain real model behavior rather than trace coincidences.

---

## 2026-04-20 — 022 — Do not treat the first WAL-LDI learned semantic ISA as a usable ISA baseline either

**Decision.** Keep `WAL with Learned Discrete ISA + Semantic Constraints (WAL-LDI)` as an explored language-design direction, but do not promote the first post-hoc semantic-ISA prototype to a new ISA baseline or runtime branch.

**Rationale.** The first WAL-LDI probe finally stopped adding wrappers on top of the flat `12 x 256` stage-id stream and tried to design a two-level ISA directly on the block-program surface. On `l54.gate/up`, `m27_wal_ldi_proto.py` learns `4` semantic high-level families per layer using a joint feature-plus-sequence clustering objective, then learns `4` low-level atoms for each of `4` slots within every family. That yields `68` explicit instructions per layer: `4` semantic family opcodes and `64` family-conditioned low-level atoms. This is the first probe in the WAL chain that creates a real semantic partition rather than just a syntactic wrapper: all four families are active, family entropy is about `1.14`, and the small `COMMON_CORE` family separates strongly (`margin ~6.36` on `gate`, `~6.11` on `up`) with noticeably better local approximate match (`~0.113 / 0.109`). But the resulting exact program surface is still not useful as an ISA baseline. Average high-level program length rises to `12.9969 / 12.9971` out of raw `12`, because every block pays a family header while low-level atoms almost never fire (`avg_low_level_calls = 0.00312 / 0.00289`, token coverage `0.000521 / 0.000483`). The approximate `ldi_only` surface is better than the first WAL-ASM approximation but still far outside the quality bar: reconstruction relMSE is `1.7697 / 1.8243` by direct compare (`eval-layer relMSE ~1.77`), and PPL degrades from `2.4081` to `2.8318` at essentially unchanged throughput and peak VRAM.

**Alternatives.** Treat the appearance of a real semantic family partition and a somewhat better approximate PPL than WAL-ASM as enough evidence that the semantic-ISA direction is already solved. Rejected for now: the first LDI pass still acts on top of an already-diffuse stage program, so it discovers only a tiny coherent island while the main language remains too flat to support useful hierarchical compression. The next viable LDI step would need semantic constraints inside the encoder or ISA-learning process itself, not only post-hoc clustering over a stage stream that is already almost maximally unique.

---

## 2026-04-20 — 023 — Do not treat the first WAL-E2E end-to-end semantic ISA as a usable ISA baseline either

**Decision.** Keep `WAL with End-to-End Learned ISA + Semantic Encoder (WAL-E2E)` as the right next language-design direction, but do not promote the first end-to-end prototype to a new ISA baseline or runtime branch.

**Rationale.** `m27_wal_e2e_proto.py` is the first probe in the WAL line that moves semantic ISA learning inside the encoder loop itself. On `l54.gate/up` it trains a small semantic encoder plus family-conditioned atom bank jointly on sampled block programs, using token reconstruction, semantic neighborhood consistency, family cohesion, anti-collapse regularization, and persistent light supervision from the initialization pass. This does land one important positive result: unlike the first post-hoc-only pipelines, the authoritative end-to-end run keeps all four families active on both layers (`active_family_count = 4`), with entropy `1.147` on `gate` and `0.802` on `up`, so semantic families are no longer only a relabeling step after the fact. A coherent `COMMON_CORE` region still exists (`~1.97%` share and margin `~6.34` on `gate`; `~7.43%` share and margin `~2.76` on `up`). But the actual program surface remains unusable. Exact program length grows to `12.9988 / 12.9988` out of raw `12`, low-level atoms fire only `0.00119 / 0.00118` times per block, and low-level token coverage drops to `0.000199 / 0.000197`, worse than the earlier post-hoc WAL-LDI structure. The approximate `e2e_only` surface also fails the quality bar: eval-layer relMSE is `~1.70`, direct layer relMSE `1.6369 / 1.7663`, and PPL degrades from `2.4081` to `2.8847` at essentially unchanged throughput and peak VRAM. This means the first end-to-end objective learns to keep semantic families alive, but still does not learn a shorter or more semantics-bearing program.

**Alternatives.** Treat the fact that semantic families survived inside the encoder loop as enough evidence that end-to-end WAL is already solved. Rejected for now: the first E2E formulation still optimizes mostly stage-token reconstruction on top of the existing `12 x 256` language, so it regularizes the labels more than it changes the program. The next viable E2E step would need direct block-reconstruction or layer-output semantics together with an explicit program-cost or correction-budget objective, not only semantic regularization around token prediction.

---

## 2026-04-20 — 024 — Do not treat the first WAL-DR direct-reconstruction objective as a usable ISA baseline yet

**Decision.** Keep `WAL with Direct Block Reconstruction + Explicit Program Cost (WAL-DR)` as the first objective-level correction that genuinely improves the approximation surface, but do not promote the first prototype to a new ISA baseline or runtime branch.

**Rationale.** `m27_wal_dr_proto.py` is the first WAL probe on `l54.gate/up` that changes the controlling objective itself rather than only the syntax or parameterization around the existing stage language. It keeps the end-to-end semantic-encoder surface from WAL-E2E (`4` families plus `64` family-conditioned low-level atoms), but replaces token-CE as the main target with direct final-block reconstruction under the current Block-RVQ surface and an explicit program-cost proxy that penalizes correction-heavy exact programs. This does produce a real empirical shift. All four families remain active (`family_entropy = 1.1188 / 1.2486`), and the approximate `dr_only` surface improves materially over both WAL-LDI and WAL-E2E: eval-layer relMSE falls to about `1.54` (`1.5051 / 1.5692` by layer), and PPL degrades only from `2.4081` to `2.7325`, much better than `2.8318` for `ldi_only` and `2.8847` for `e2e_only`. But the learned program surface is still not compressible enough to be treated as a new ISA baseline. Exact program length remains `12.9984 / 12.9988` out of raw `12`, low-level atoms still fire only `0.00156 / 0.00117` times per block, and low-level token coverage is still only `0.000260 / 0.000195`, so the exact program is still overwhelmingly correction-dominated.

**Alternatives.** Treat the improved approximate PPL/relMSE as enough evidence that the ISA problem is basically solved once the objective is fixed. Rejected for now: the first WAL-DR prototype proves that objective choice matters, but it improves semantics quality much more than it improves compression of the exact program. The next viable WAL-DR step would need to couple cost even more directly to downstream behavior, for example through layer-output or activation-conditioned reconstruction and/or a more explicit learned decision about when to invoke high-level calls versus paying literal corrections.

---

## 2026-04-20 — 025 — Do not treat the first WAL-LO layer-output objective as a usable ISA baseline either

**Decision.** Keep `WAL with Layer-Output Reconstruction + Strong Program Cost (WAL-LO)` as the necessary next objective-level probe after WAL-DR, but do not promote the first activation-conditioned layer-output prototype to a new ISA baseline or runtime branch.

**Rationale.** `m27_wal_lo_proto.py` moved the main loss from direct block reconstruction to activation-conditioned layer-output contribution reconstruction on real captured inputs from the target layer, while also adding stronger exact-program cost terms (per-literal penalty, overlength hinge, too-few-atoms penalty) and an explicit atom-vs-literal decision head. The exact path stays correct by construction. However the new surface does not beat WAL-DR. The approximate `lo_only` path degrades from `PPL = 2.4081` to `2.7463`, slightly worse than `dr_only = 2.7325`. More importantly, the exact program still collapses to almost full literal fallback: average program length is `13.0018 / 13.0000` out of raw `12`, average low-level calls only `0.0140 / 0.0000`, and low-level token coverage only `0.00102 / 0.00000`. The explicit decision head therefore learns the honest but unwanted economic verdict that atoms are usually not worth using.

**Alternatives.** Treat the stronger downstream proxy as enough evidence that the direction is already solved, or keep tuning the same WAL-LO surface as if this were only a mild hyperparameter miss. Rejected for now: the first honest gate shows a deeper structural issue. Moving the loss closer to downstream layer behavior is still not sufficient when the atom bank itself does not become economically useful for the exact program. The next viable step would need to improve exact atom usefulness directly, not only supply a better downstream proxy.

---

## 2026-04-20 — 026 — Do not treat the first WAL-LHA expressive-atom surface as a usable ISA baseline either

**Decision.** Keep `WAL with Learnable High-Expressivity Atoms (WAL-LHA)` as the correct next probe after WAL-LO, but do not promote the first expressive-atom prototype to a new ISA baseline or runtime branch.

**Rationale.** `m27_wal_lha_proto.py` kept the activation-conditioned layer-output objective from WAL-LO, but changed the low-level atom surface itself: each atom now acts as a tiny expressive module that produces context-dependent token logits on top of a learned base phrase prior, and training adds a separate atom-selection loss plus a direct atom-usage bonus when the expressive atom beats the static bank. This does produce a real positive shift on the approximate side. The `lha_only` path degrades only from `PPL = 2.4081` to `2.7179`, which is better than WAL-LO (`2.7463`) and slightly better than WAL-DR (`2.7325`), while sample layer-output relMSE also improves materially (`1.3418 / 1.2052`). But the exact program still does not become economically useful. Average program length remains `13.0006 / 13.0000` out of raw `12`, average low-level calls are only `0.0083 / 0.0000`, and exact fallback remains overwhelmingly literal. So the first expressive-atom probe improves approximate semantics without making atoms profitable as exact instructions.

**Alternatives.** Treat the improved approximate PPL/relMSE as enough evidence that expressive atoms already solve the ISA problem, or keep iterating this exact surface as if only atom capacity were missing. Rejected for now: the first honest gate shows that stronger atom expressivity helps the approximate language much more than the exact program. The next viable step would need to improve the projection of expressive atoms into economically useful exact instructions, not only make them better approximate modules.

---

## 2026-04-20 — 027 — Reinterpret exactness through strict budgets before attempting WAL-CDA

**Decision.** Keep `WAL with Strict Budgeted Compatibility (WAL-SBC)` as the next probe after WAL-LHA and defer `WAL-CDA` until budgeted exactness shows non-trivial program-economics gain.

**Rationale.** `m27_wal_sbc_proto.py` reuses the expressive atoms from WAL-LHA, but changes the exact contract itself. The old strict legacy path remains available, while the new `budgeted_exact` path accepts an atom only if both block-level and activation-conditioned output relMSE stay within strict budgets; otherwise it falls back first to a small residual phrase bank and only then to literals. This first honest `16`-window gate shows that the contract shift is safe but not yet sufficient. `strict_legacy` gives `PPL = 2.8055`, `1001.80 tok/s`, peak `69447.3 MB`, while `budgeted_exact` gives `PPL = 2.8069`, `1001.63 tok/s`, with the same peak memory. So revising exactness does not catastrophically break quality or runtime. But the structural surface barely moves: average program length remains `12.9983 / 12.9985` out of raw `12`, atom calls are effectively zero (`2.9e-06 / 5.7e-06`), residual calls are only `0.00172 / 0.00144`, and accepted-budget token coverage stays around `4e-4`. In other words, the first strict budget geometry is still too conservative to buy meaningful exact-program usage.

**Alternatives.** Jump straight to `WAL-CDA` and make the atoms even more context-aware. Rejected for now: WAL-LHA already showed that stronger atom expressivity improves approximate semantics much more than exact economics. The next viable move is to tune budget geometry and residual fallback economics first, then revisit more dynamic atoms only if this contract starts to purchase real program shortening.

---

## 2026-04-20 — 028 — Use empirical SBC profiling to falsify loose percentile budgets before committing to WAL-CDA

**Decision.** Run the WAL-SBC follow-up in two steps: first an offline empirical budget profiler over the saved expressive-atom ISA, then a narrow targeted sweep on percentile-derived budgets with `4`-window screening and a `16`-window honest gate only for the best two configs. Treat raw percentile budgets as rejected for the exactness contract if they buy strong non-literal usage but clearly break quality.

**Rationale.** `m27_wal_sbc_budget_profile.py` confirmed that the first strict SBC budgets were far below the observed atom/residual error geometry. On `l54.gate/up`, atom output relMSE lives around `p50 ~ 1.30 / 1.32`, `p70 ~ 1.57 / 1.62`, `p80 ~ 1.76 / 1.82`, `p90 ~ 2.06 / 2.18`, while the original atom budget was only `0.05`; the residual output relMSE distribution is also centered near `0.82 - 0.85` rather than `0.16`. So the original deadness really did come from an over-tight contract, not from a missing non-literal surface.

`m27_wal_sbc_tune_proto.py` then ran exactly the narrow grid derived from those empirical quantiles: `atom_output_budget in {p70,p80,p90}`, `residual_output_budget in {p85,p92}`, `residual_program_cost in {0.5,0.75,1.0}`, with screening on `4` windows and honest promotion of only the top two configs to the `16`-window gate. This produced the opposite failure mode from the first WAL-SBC prototype. Non-literal usage becomes strong immediately: the best finalists reach `mean_nonliteral_calls ~ 3.85` and `avg_program_length ~ 4.78 / 4.90`, far below raw `12`. But the same configs collapse quality: `4`-window `PPL delta vs strict legacy` is already about `+0.284 / +0.295`, and the honest `16`-window gate worsens to `+0.368 / +0.380` (`PPL ~ 3.17 - 3.19` vs `2.8055` strict legacy).

So the question is no longer whether budget tuning can turn on the non-literal path. It can. The clean result is that raw empirical percentile CDFs are too loose to be the exactness contract: they purchase a short program only by allowing too much semantic drift.

**Alternatives.** Continue blind SBC sweeps in the same percentile regime. Rejected: the new two-step profiler+tune already spans the intended empirical zone and shows a stable quality failure, not an isolated bad point. If this line continues before `WAL-CDA`, the next move must be a much tighter or more shape-aware acceptance surface, not more blind percentile widening.

---

## 2026-04-20 — 029 — Move from WAL-SBC tuning to economically constrained WAL-CDA

**Decision.** Treat `WAL-CDA` as the next justified probe after WAL-SBC Step 12e. Do not frame it as a more powerful unconstrained atom generator. The prototype should be an economically constrained context-aware atom layer whose main job is to improve the basis itself while preserving the exactness comparison surface.

**Rationale.** Step 12e made the diagnosis clean. The percentile-driven SBC tune proved that the current atom/residual basis can purchase a short program if the contract is loosened enough: the best finalists drive `avg_program_length` down to about `4.78 - 4.90` and lift mean non-literal calls to about `3.85 - 3.88`. So the non-literal path is not inherently dead. But the same configs break the sacred quality bar badly: the honest `16`-window gate degrades by about `+0.368 / +0.380` PPL vs strict legacy and pushes average layer relMSE to about `1.04 - 1.12`. That is no longer a threshold-tuning problem. It is evidence that the current atoms/residual proposals are too coarse to support economically useful exactness.

So the next move must change the basis, not just the budgets. But the Step 12e failure also constrains how to do that. `WAL-CDA` should be built as `base atom prior + context-conditioned delta`, not as a free generative decoder. Its new training signal should reward exact usefulness under budget, not only reconstruction similarity. And evaluation must preserve three explicit surfaces: `strict_legacy`, the old `budgeted_exact` basis, and `budgeted_exact + CDA`, so that any gain can be attributed to the new basis rather than a changed acceptance contract.

The implementation guardrails are concrete, not only conceptual. Discrete IDs should enter through embeddings rather than raw numeric coordinates, the context input should be a cheap detached activation summary rather than the full live layer input, the delta path should stay factorized and low-rank (`alpha(h) @ shared_basis`) with an explicit norm/temperature cap, and the usefulness objective should be a soft expected-cost surrogate under the same budget contract. That surrogate must compare CDA against the best old accepted path on the existing budgeted basis, not only against strict legacy logits.

**Alternatives.** Keep widening or reshaping SBC budgets before changing the basis. Rejected as the default next move: Step 12e already proved that loose budgets can buy economics but not quality, so another generic SBC sweep would mostly revisit a diagnosed failure mode. Any future SBC work would need a much tighter or more shape-aware acceptance geometry, but the main frontier question is now the basis itself.


---

## 2026-04-20 — 030 — WAL-0 as the base ISA for weight language

**Decision.** Adopt WAL-0 (scalar k-means atoms + greedy residual + ternary signs) as the base ISA for the weight language, deferring WAL-1 (vector atoms) until coefficient representation is solved.

**Rationale.** M45-M53 established that WAL-0 scalar achieves PPL 2.7858 on Llama 3.3 70B with only K=128, lmax=2 — essentially dense quality (+0.19%). WAL-1 vector atoms failed catastrophically (relMSE 0.08-0.99) because ternary {-1,0,+1} with lmax=2 cannot represent high-dimensional vectors. The scalar ISA is therefore the only proven foundation. The language layer (codebook, variable length, grammar) should be built on top of this proven ISA rather than on an unproven vector foundation.

**Alternatives.** 
- Continue WAL-1 with continuous coefficients (4-bit) or lmax≥8. Rejected for now: would explode decode complexity and runtime before the language layer exists.
- Return to DRL v2 ternary routes. Rejected: WAL-0 beats DRL v2 by 200× on relMSE at same K.
- Fixed ladder (no k-means). Rejected: k-means atoms are data-adaptive and give better coverage.

**What WAL-0 needs to become a language.**
1. Codebook layer: unique programs → IDs (M54).
2. Variable length: stop_depth per weight (M55).
3. Grammar: reusable patterns in program stream (M56).

**What WAL-0 explicitly does NOT solve yet.**
- Compression ratio: 2 bytes/weight is too high for shipping. Codebook + variable length + Huffman/ANS must reduce this.
- Hierarchical structure: scalar atoms are flat. Vector/tensor atoms (WAL-1+) remain the long-term direction once coefficient representation is solved.

---

## 2026-04-20 — 031 — GPU-native everything for WAL-0 operations

**Decision.** All WAL-0 encode, decode, codebook mining, and analysis operations must stay on GPU. No CPU copies of programs, codes, or atom tables during batch operations.

**Rationale.** M53c v1 failed because `wal_encode_scalar` copied programs to CPU. M53c v3 showed that even `fast_encode` (GPU-native PyTorch) is still slow because of Python loop overhead. The fused Triton kernel (M53b) eliminated this. For codebook mining (M54), the same principle applies: `torch.unique` on GPU tensors with 16M elements is fast; copying to CPU for analysis is slow and memory-intensive. GPU-native is the default; CPU fallback is only for final serialization to disk.

**Alternatives.** 
- CPU-backed codebook analysis (M53 OOM fix). Rejected as default: it was a workaround for OOM during `torch.unique` on full tensors, not the desired architecture. The fix is to sample on GPU or use `torch.unique` with smaller chunks, not to move to CPU.
- NumPy/SciPy for frequency analysis. Rejected: PyTorch GPU ops are sufficient for unigram/bigram/entropy calculations.


---

## 2026-04-20 — 011 — RelMSE is permanently discarded as a quality metric

**Decision.** Only full-model PPL is valid. relMSE, sign agreement, output correlation, and all single-layer metrics are unreliable predictors.

**Rationale.** M43 showed VRE with relMSE 0.001 and output correlation 0.9992 is catastrophically toxic to full-model PPL (>7000). M71 showed single-layer PPL can be +0.0002 while full-model PPL is +4.90 (difference up to 24,500×). The user explicitly directed: "МОЖЕШЬ ЗАБЫТЬ ПРО relMSE ВООБЩЕ".

**Alternatives.** Keep relMSE as a diagnostic. Rejected: it creates false confidence and has historically misled every compression prototype.

---

## 2026-04-20 — 012 — 12 bits/weight is the hard floor for 70B Llama

**Decision.** Empirically, anything less than 12 bits/weight causes catastrophic PPL degradation. This is treated as a physical constraint, not a temporary limitation.

**Rationale.** M69-M73 systematic sweep: K=16→111k PPL, K=32→1.5k, K=64→46.6, K=128→7.68, K=256→3.02 DEGRADE. Two-tier PQ at 8 bits: 3.1137 PPL. Position-specific, SVD, tile/vector — all fail at <12 bits. The degradation is caused by structured error accumulation across 80 layers, not by any single layer's inaccuracy.

**Alternatives.** Smaller blocks, learned coeff tables, adaptive precision. Rejected: all tested in M65-M74 and all fail the full PPL gate.

---

## 2026-04-20 — 013 — WAL v1 is a semantic layer, not a compression improvement

**Decision.** WAL v1 (hierarchical atoms) adds interpretability and language structure on top of WAL v2. It does not improve compression. Same 12 bits/weight, same PPL quality.

**Rationale.** M75 proved hierarchical atoms achieve PPL 2.7809 (delta +0.0004 PASS) — identical to WAL v2. 35,840 L1 atoms were created but programs remain 12 bits/weight. The value is in interpretability: atoms have recursive definitions enabling semantic analysis, program tracing, and human-readable explanations.

**Alternatives.** Pursue vector atoms, tensor blocks, or other compression schemes. Rejected: empirically proven impossible to beat 12-bit floor without quality loss.

---

## 2026-04-20 — 014 — Binary format v1 must support hierarchical atom definitions

**Decision.** WAL v1 binary format includes serialized hierarchical definitions alongside base atoms and programs.

**Rationale.** Hierarchical atoms are a core feature of WAL v1. The binary format must round-trip them exactly. Format v1.0 includes: op codes (ADD/MUL/NEG/CONST), child IDs, and coeffs per L1+ definition.

**Alternatives.** Store only precomputed flat atoms. Rejected: loses interpretability — the whole point of WAL v1.

---

## 2026-04-20 — 015 — PyTorch integration uses lazy decode with optional cache

**Decision.** WALParameter decodes on demand and supports both on-the-fly (WALLinear) and persistent-cache (WALCachedLinear) modes.

**Rationale.** Full decode of 70B model to bf16 costs ~140GB VRAM. Lazy decode lets users trade memory for speed. WALLinear decodes per forward pass (slow, low memory). WALCachedLinear decodes once and keeps dense weights (fast, high memory). User chooses based on constraints.

**Alternatives.** Always decode to dense on load. Rejected: would defeat the purpose of compression.

---

## 2026-04-20 — 016 — Unit8 tensor indexing requires explicit int() conversion

**Decision.** When indexing tensors with uint8 scalar values, always convert to int() first.

**Rationale.** PyTorch treats uint8 indexing as boolean mask selection, not integer indexing. `coeffs[prog.coeff_ids[i]]` where coeff_ids[i] is a uint8 scalar tensor returns a tensor of shape [8] (all values where mask bit is set), not a single scalar. This caused multiple bugs in decoder.py and asm.py.

**Alternatives.** Use .long() or .item() consistently. Accepted: int() is concise and correct for scalar values.
