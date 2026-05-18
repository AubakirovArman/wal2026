"""
M9c: Activation sparsity probe for `down_proj` inputs (post-SwiGLU).

Question: how much of `x` going into down_proj is near-zero?
If a meaningful fraction of K-tiles have max|x| below threshold, we can
skip decode + GEMM for those tiles -> direct speedup with zero quality loss
(at conservative thresholds).

Method:
  1. Load Llama-3.3-70B with EagerBf16Linear (baseline quality).
  2. Hook all `mlp.down_proj` modules to capture incoming `x` (the SiLU(gate)*up tensor).
  3. Run 4 WikiText-2 windows (2048 tokens each).
  4. For each captured x of shape [tokens, K], compute:
     - per-token max|x| (row-wise max).
     - tile-wise max|x| with TILE in {32, 64, 128} along K dim.
     - fraction of TILES below absolute threshold 1e-4, 1e-3, 1e-2.
     - fraction of TILES below RELATIVE threshold (vs row max).
  5. Report mean and per-layer-bucket (early/mid/late) statistics.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit")

from dwl2_dynamic_route.src.runtime import replace_with_eager_bf16

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
DATA_PATH = "/mnt/hf_model_weights/arman/3bit/bk/wikitext2_test.txt"
NUM_WINDOWS = 4
MAX_LEN = 2048

# Aggregation buckets: store sums + counts so we can print mean fractions
TILES = (32, 64, 128)
ABS_THR = (1e-4, 1e-3, 1e-2)
REL_THR = (1e-3, 1e-2, 1e-1)


class TileStats:
    __slots__ = ("n_tiles_total", "n_below_abs", "n_below_rel", "max_overall", "mean_max")

    def __init__(self):
        self.n_tiles_total = 0
        self.n_below_abs = {t: {a: 0 for a in ABS_THR} for t in TILES}
        self.n_below_rel = {t: {r: 0 for r in REL_THR} for t in TILES}
        self.max_overall = 0.0
        self.mean_max = 0.0  # running mean of per-token max
        self.token_count = 0

    def update(self, x: torch.Tensor) -> None:
        # x: [T, K] float (any dtype)
        x = x.detach().float().abs()
        T, K = x.shape
        self.token_count += T
        per_tok_max = x.max(dim=1).values  # [T]
        self.mean_max = (self.mean_max * (self.token_count - T) + per_tok_max.sum().item()) / self.token_count
        self.max_overall = max(self.max_overall, per_tok_max.max().item())
        row_max = per_tok_max.clamp_min(1e-12)  # [T]
        for tile in TILES:
            if K % tile != 0:
                pad = tile - (K % tile)
                xp = torch.nn.functional.pad(x, (0, pad), value=0.0)
            else:
                xp = x
            Tn = xp.shape[0]
            Kp = xp.shape[1]
            tile_max = xp.view(Tn, Kp // tile, tile).max(dim=-1).values  # [T, Kp/tile]
            self.n_tiles_total += int(tile_max.numel())
            for a in ABS_THR:
                self.n_below_abs[tile][a] += int((tile_max < a).sum().item())
            rel = tile_max / row_max[:, None]
            for r in REL_THR:
                self.n_below_rel[tile][r] += int((rel < r).sum().item())

    def to_dict(self) -> dict:
        out = {
            "n_tiles_total": self.n_tiles_total,
            "max_overall": self.max_overall,
            "mean_per_tok_max": self.mean_max,
            "frac_below_abs": {},
            "frac_below_rel": {},
        }
        for tile in TILES:
            denom_per_tile = self.n_tiles_total // (1 if tile == TILES[0] else 1)  # rough: separate per tile
            # actually we collected n_tiles_total summed across tiles -> recompute denom per tile
            # we need per-tile denom; easier: derive from token_count
        # Recompute denominators per tile from token_count
        for tile in TILES:
            kp_over_tile = (8192 + tile - 1) // tile  # gate output dim is 8192? down_proj input = 28672
            # We don't know K here exactly; recompute outside.
            pass
        return out  # we'll recompute properly outside


def main() -> None:
    print("loading model...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  load took {time.time()-t0:.0f}s", flush=True)

    # Replace with eager-bf16 (baseline-quality runtime, materialized weights)
    print("encoding to route+materialize...", flush=True)
    t1 = time.time()
    replace_with_eager_bf16(model)
    print(f"  replace took {time.time()-t1:.0f}s", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

    # Per-layer stats containers
    layer_stats: dict[int, dict] = {}
    K_per_layer: dict[int, int] = {}

    hooks = []

    def make_hook(layer_idx: int):
        def hook(_mod, inputs, _out):
            x = inputs[0]
            if x.dim() == 3:
                T = x.shape[0] * x.shape[1]
                x2 = x.reshape(T, x.shape[-1])
            else:
                x2 = x
            K = int(x2.shape[1])
            K_per_layer[layer_idx] = K
            st = layer_stats.setdefault(layer_idx, _new_stats())
            _update_stats(st, x2)
        return hook

    def _new_stats() -> dict:
        return {
            "tokens": 0,
            "n_tiles": {t: 0 for t in TILES},
            "n_below_abs": {t: {a: 0 for a in ABS_THR} for t in TILES},
            "n_below_rel": {t: {r: 0 for r in REL_THR} for t in TILES},
            "sum_per_tok_max": 0.0,
            "max_overall": 0.0,
        }

    def _update_stats(st: dict, x: torch.Tensor) -> None:
        x = x.detach().float().abs()
        T, K = x.shape
        st["tokens"] += T
        per_tok_max = x.max(dim=1).values
        st["sum_per_tok_max"] += float(per_tok_max.sum().item())
        st["max_overall"] = max(st["max_overall"], float(per_tok_max.max().item()))
        row_max = per_tok_max.clamp_min(1e-12)
        for tile in TILES:
            Kp = ((K + tile - 1) // tile) * tile
            if Kp != K:
                xp = torch.nn.functional.pad(x, (0, Kp - K), value=0.0)
            else:
                xp = x
            tile_max = xp.view(T, Kp // tile, tile).max(dim=-1).values  # [T, n_tiles]
            n = int(tile_max.numel())
            st["n_tiles"][tile] += n
            for a in ABS_THR:
                st["n_below_abs"][tile][a] += int((tile_max < a).sum().item())
            rel = tile_max / row_max[:, None]
            for r in REL_THR:
                st["n_below_rel"][tile][r] += int((rel < r).sum().item())

    # Attach hooks
    n_layers = len(model.model.layers)
    for i in range(n_layers):
        h = model.model.layers[i].mlp.down_proj.register_forward_hook(make_hook(i))
        hooks.append(h)
    print(f"  attached {len(hooks)} hooks on down_proj", flush=True)

    # Load text & tokenize
    text = Path(DATA_PATH).read_text(errors="ignore")
    toks = tok(text, return_tensors="pt").input_ids[0]
    print(f"  tokens: {toks.numel()}", flush=True)

    # Run NUM_WINDOWS windows
    device = next(model.parameters()).device
    with torch.inference_mode():
        for w in range(NUM_WINDOWS):
            start = w * MAX_LEN
            end = start + MAX_LEN
            if end > toks.numel():
                break
            window = toks[start:end].unsqueeze(0).to(device)
            print(f"  window {w+1}/{NUM_WINDOWS}: {window.shape}", flush=True)
            _ = model(window)

    for h in hooks:
        h.remove()

    # Aggregate to JSON-friendly
    out = {"per_layer": {}, "config": {"tiles": list(TILES), "abs_thr": list(ABS_THR), "rel_thr": list(REL_THR), "num_windows": NUM_WINDOWS, "max_len": MAX_LEN}}
    for i, st in sorted(layer_stats.items()):
        K = K_per_layer.get(i, -1)
        rec = {"K": K, "tokens": st["tokens"], "max_overall": st["max_overall"],
               "mean_per_tok_max": st["sum_per_tok_max"] / max(st["tokens"], 1),
               "frac_below_abs": {}, "frac_below_rel": {}}
        for tile in TILES:
            denom = max(st["n_tiles"][tile], 1)
            rec["frac_below_abs"][str(tile)] = {f"{a:.0e}": st["n_below_abs"][tile][a] / denom for a in ABS_THR}
            rec["frac_below_rel"][str(tile)] = {f"{r:.0e}": st["n_below_rel"][tile][r] / denom for r in REL_THR}
        out["per_layer"][str(i)] = rec

    out_path = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route/results/m9c_act_sparsity_down_proj.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Print summary: mean across layer-buckets
    print("\n=== SUMMARY (fraction of K-tiles where max|x| < threshold) ===")
    print(f"  buckets: early=0-26  mid=27-52  late=53-79  ({n_layers} total layers)")
    for tile in TILES:
        print(f"\n  TILE={tile}")
        for thr_kind, thr_list in (("abs", ABS_THR), ("rel", REL_THR)):
            print(f"    {thr_kind:>3s}_thr: " + "  ".join(f"{x:>8.0e}" for x in thr_list))
            for label, lo, hi in (("early", 0, 26), ("mid", 27, 52), ("late", 53, 79)):
                vals = []
                for thr in thr_list:
                    fracs = []
                    for i in range(lo, min(hi+1, n_layers)):
                        rec = out["per_layer"].get(str(i))
                        if rec is None:
                            continue
                        kind = "frac_below_abs" if thr_kind == "abs" else "frac_below_rel"
                        key = f"{thr:.0e}"
                        fracs.append(rec[kind][str(tile)][key])
                    vals.append(sum(fracs) / max(len(fracs), 1))
                print(f"    {label:>5s}:  " + "  ".join(f"{v:>8.4f}" for v in vals))


if __name__ == "__main__":
    main()
