"""M27 PTDP Step 2a: per-tile dynamic palette offline gate.

For ``model.language_model.layers.54.mlp.gate_proj`` and ``model.language_model.layers.54.mlp.up_proj`` this
script computes, for each tile of shape ``row_tile_size x col_tile_size``:

  1. tile-local activation-weighted influence mass per ``(stage, id)``
  2. tile-local occurrence counts per ``(stage, id)``
  3. PTDP hit rate after selecting top-k ids by tile-local influence

The goal is to falsify or support the PTDP hypothesis before any kernel work.
If tile-local top-k still does not reach high coverage, there is no reason to
integrate a per-tile palette runtime branch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from dwl2_dynamic_route.src.runtime import (  # noqa: E402
    PackedGroupedBlockRVQLinear,
    replace_with_packed_block_rvq,
)

MODEL_DIR = WORKSPACE_ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TEXT_PATH = WORKSPACE_ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512
DEFAULT_TARGETS = (
    "model.language_model.layers.54.mlp.gate_proj",
    "model.language_model.layers.54.mlp.up_proj",
)


def _eval_ids(tok: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


class _TilePaletteHook:
    def __init__(self, module: PackedGroupedBlockRVQLinear, row_tile_size: int, col_tile_size: int) -> None:
        if len(module.groups) != 1:
            raise ValueError("PTDP Step 2a expects a single packed group for l54.gate/up")
        self.module = module
        self.group = module.groups[0]
        self.row_tile_size = int(row_tile_size)
        self.col_tile_size = int(col_tile_size)
        self.block_size = int(self.group.block_size)
        if self.col_tile_size % self.block_size != 0:
            raise ValueError(f"col_tile_size={col_tile_size} must be divisible by block_size={self.block_size}")
        self.block_cols_per_tile = self.col_tile_size // self.block_size
        self.n_size = int(getattr(self.group, "stage_ids_0").shape[0])
        self.blocks_per_row = int(getattr(self.group, "stage_ids_0").shape[1])
        self.num_row_tiles = math.ceil(self.n_size / self.row_tile_size)
        self.num_col_tiles = math.ceil(self.blocks_per_row / self.block_cols_per_tile)
        self.num_tiles = self.num_row_tiles * self.num_col_tiles
        self.num_stages = int(self.group.num_stages)
        self.M = int(getattr(self.group, "codebook_0").shape[0])
        self.row_scale = self.group.row_scale.float().abs().cpu()
        self.stage_ids = [
            getattr(self.group, f"stage_ids_{stage_idx}").to(torch.int64).cpu().contiguous()
            for stage_idx in range(self.num_stages)
        ]
        self.codebook_norms = []
        for stage_idx in range(self.num_stages):
            cb = getattr(self.group, f"codebook_{stage_idx}").float()
            if self.group.stage_scales is not None:
                cb = cb * self.group.stage_scales[stage_idx].float()
            self.codebook_norms.append(cb.norm(dim=-1).cpu().contiguous())
        self.stage_tile_counts = [torch.zeros((self.num_tiles, self.M), dtype=torch.float32) for _ in range(self.num_stages)]
        self.stage_tile_influence = [torch.zeros((self.num_tiles, self.M), dtype=torch.float32) for _ in range(self.num_stages)]

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_flat = x.reshape(-1, x.shape[-1])
        pad_cols = int(self.group.padded_cols) - int(self.group.in_features)
        x_padded = F.pad(x_flat, (0, pad_cols)) if pad_cols > 0 else x_flat
        x_blocks = x_padded.view(x_flat.shape[0], self.blocks_per_row, self.block_size)
        block_energy = x_blocks.float().norm(dim=-1).mean(dim=0).cpu()
        for row_tile_idx in range(self.num_row_tiles):
            row_a = row_tile_idx * self.row_tile_size
            row_b = min(row_a + self.row_tile_size, self.n_size)
            row_scale_tile = self.row_scale[row_a:row_b]
            for col_tile_idx in range(self.num_col_tiles):
                block_a = col_tile_idx * self.block_cols_per_tile
                block_b = min(block_a + self.block_cols_per_tile, self.blocks_per_row)
                block_energy_tile = block_energy[block_a:block_b]
                tile_idx = row_tile_idx * self.num_col_tiles + col_tile_idx
                for stage_idx in range(self.num_stages):
                    ids = self.stage_ids[stage_idx][row_a:row_b, block_a:block_b].reshape(-1).contiguous()
                    counts = torch.bincount(ids, minlength=self.M).to(torch.float32)
                    weights = (
                        row_scale_tile.repeat_interleave(block_b - block_a)
                        * block_energy_tile.repeat(row_b - row_a)
                        * self.codebook_norms[stage_idx].index_select(0, ids)
                    )
                    influence = torch.bincount(ids, weights=weights, minlength=self.M)
                    self.stage_tile_counts[stage_idx][tile_idx].add_(counts)
                    self.stage_tile_influence[stage_idx][tile_idx].add_(influence)


def _summarise_tiles(
    counts: torch.Tensor,
    influence: torch.Tensor,
    topk_values: list[int],
) -> list[dict]:
    rows: list[dict] = []
    total_counts = counts.sum(dim=1).clamp_min(1.0)
    total_influence = influence.sum(dim=1).clamp_min(1e-12)
    for topk in topk_values:
        k = min(int(topk), int(influence.shape[1]))
        hot = torch.topk(influence, k=k, dim=1).indices
        count_hit = counts.gather(1, hot).sum(dim=1) / total_counts
        influence_hit = influence.gather(1, hot).sum(dim=1) / total_influence
        rows.append({
            "topk": k,
            "mean_count_hit": float(count_hit.mean()),
            "median_count_hit": float(count_hit.median()),
            "min_count_hit": float(count_hit.min()),
            "mean_influence_hit": float(influence_hit.mean()),
            "median_influence_hit": float(influence_hit.median()),
            "min_influence_hit": float(influence_hit.min()),
            "frac_count_hit_ge_075": float((count_hit >= 0.75).float().mean()),
            "frac_count_hit_ge_085": float((count_hit >= 0.85).float().mean()),
            "frac_influence_hit_ge_075": float((influence_hit >= 0.75).float().mean()),
            "frac_influence_hit_ge_085": float((influence_hit >= 0.85).float().mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--num-windows", type=int, default=2)
    ap.add_argument("--text-source", choices=("raw", "local"), default="raw")
    ap.add_argument("--group-rows", type=int, default=28672)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    ap.add_argument("--row-tile-size", type=int, default=64)
    ap.add_argument("--col-tile-size", type=int, default=256)
    ap.add_argument("--topk", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--out", default=str(REPO_ROOT / "results/m23_influence/l54_gate_up_ptdp.pt"))
    ap.add_argument("--out-summary", default=str(REPO_ROOT / "results/m23_influence/l54_gate_up_ptdp_summary.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    print("[load] model ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    print("[encode] replacing target layers with packed Block-RVQ ...", flush=True)
    t0 = time.time()
    replace_with_packed_block_rvq(
        model,
        tuple(args.targets),
        group_rows=args.group_rows,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        num_stages=args.num_stages,
        product_splits=args.product_splits,
        calibrate_stage_scales=True,
        matmul_strategy="full_weight_fast",
    )
    print(f"[encode] done in {time.time() - t0:.1f}s", flush=True)

    trackers: dict[str, _TilePaletteHook] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, PackedGroupedBlockRVQLinear) and name in args.targets:
            trackers[name] = _TilePaletteHook(mod, args.row_tile_size, args.col_tile_size)

            def _make(layer_name: str):
                def _hk(_mod, args_in):
                    trackers[layer_name].update(args_in[0].detach())

                return _hk

            hooks.append(mod.register_forward_pre_hook(_make(name)))

    device = model.get_input_embeddings().weight.device
    total_len = ids.size(1)
    num_windows = min(args.num_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    print(f"[collect] windows={num_windows}", flush=True)
    with torch.no_grad():
        for idx in range(num_windows):
            begin = idx * STRIDE
            end = min(begin + MAX_LEN, total_len)
            model(ids[:, begin:end].to(device))
            print(f"  {idx + 1}/{num_windows}", flush=True)
    for hook in hooks:
        hook.remove()

    payload: dict = {"args": vars(args)}
    summary_rows: list[dict] = []
    print()
    print(f"{'layer':50s} {'stage':>5s} {'topk':>4s} {'count_hit':>10s} {'infl_hit':>9s} {'>=0.75':>8s} {'>=0.85':>8s}")
    for name, tracker in trackers.items():
        payload[name] = {
            "M": tracker.M,
            "num_stages": tracker.num_stages,
            "row_tile_size": tracker.row_tile_size,
            "col_tile_size": tracker.col_tile_size,
            "block_cols_per_tile": tracker.block_cols_per_tile,
            "num_row_tiles": tracker.num_row_tiles,
            "num_col_tiles": tracker.num_col_tiles,
            "num_tiles": tracker.num_tiles,
            "stage_tile_counts": tracker.stage_tile_counts,
            "stage_tile_influence": tracker.stage_tile_influence,
        }
        for stage_idx in range(tracker.num_stages):
            rows = _summarise_tiles(
                tracker.stage_tile_counts[stage_idx],
                tracker.stage_tile_influence[stage_idx],
                list(args.topk),
            )
            for row in rows:
                row.update({"layer": name, "stage": stage_idx})
                summary_rows.append(row)
                print(
                    f"{name:50s} {stage_idx:>5d} {row['topk']:>4d} "
                    f"{row['mean_count_hit']:>10.3f} {row['mean_influence_hit']:>9.3f} "
                    f"{row['frac_count_hit_ge_075']:>8.3f} {row['frac_count_hit_ge_085']:>8.3f}"
                )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    Path(args.out_summary).write_text(json.dumps({"args": vars(args), "rows": summary_rows}, indent=2))
    print()
    print(f"[save] {args.out}")
    print(f"[save] {args.out_summary}")


if __name__ == "__main__":
    main()