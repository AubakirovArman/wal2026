"""M27 RRF Step 1a: collect per-(stage,id) influence + structural interference.

Targets only ``model.language_model.layers.54.mlp.gate_proj`` and ``model.language_model.layers.54.mlp.up_proj``
(the same layers where M26-B2/B3 narrow gates failed) and persists the raw
tensors required by the offline Route-Register-File allocator.

Persisted artifacts (in ``results/m23_influence/l54_gate_up.pt``):

    {
        "<layer_name>": {
            "M": int,                                        # codebook size per stage
            "num_stages": int,
            "tile_size": int,                                # block_n used for interference
            "num_tiles": int,
            "n_size": int,                                   # output rows
            "blocks_per_row": int,
            "stage_influence": list[Tensor[M] (float64, cpu)],
            "stage_counts":    list[Tensor[M] (float64, cpu)],
            "stage_interference": list[Tensor[M, M] (float32, cpu)],
        },
        ...
        "args": {...},
    }

Influence is the same accumulation as M23 (activation-weighted).
Interference is **structural** — it depends only on the static stage_ids and the
register allocator's tile size (``block_n=256`` to match the B2 Triton kernel).
For each tile of consecutive output rows we collect, per stage, the set of
unique codebook ids used inside that tile; pairwise co-occurrence over tiles is
the interference matrix.

This script is intentionally tiny and read-only: it never modifies the runtime
or strategy code.
"""
from __future__ import annotations

import argparse
import gc
import json
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


class _InfluenceHook:
    """Activation-weighted influence + occurrence counts (mirrors M23.update)."""

    def __init__(self, module: PackedGroupedBlockRVQLinear) -> None:
        self.module = module
        # Layers used here are not grouped (single PackedBlockRVQGroup expected),
        # but we support the general case anyway.
        self.stage_influence: list[torch.Tensor] = []
        self.stage_counts: list[torch.Tensor] = []
        self._codebook_norms: list[torch.Tensor] = []
        for group in module.groups:
            for stage_idx in range(group.num_stages):
                while len(self.stage_influence) <= stage_idx:
                    cb = getattr(group, f"codebook_{stage_idx}")
                    self.stage_influence.append(torch.zeros(int(cb.shape[0]), dtype=torch.float64, device=cb.device))
                    self.stage_counts.append(torch.zeros(int(cb.shape[0]), dtype=torch.float64, device=cb.device))
                    self._codebook_norms.append(torch.empty(0, device=cb.device))
                if self._codebook_norms[stage_idx].numel() == 0:
                    cb = getattr(group, f"codebook_{stage_idx}").float()
                    if group.stage_scales is not None:
                        cb = cb * group.stage_scales[stage_idx].float()
                    self._codebook_norms[stage_idx] = cb.norm(dim=-1).contiguous()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_flat = x.reshape(-1, x.shape[-1])
        for group in self.module.groups:
            pad_cols = int(group.padded_cols) - int(group.in_features)
            x_padded = F.pad(x_flat, (0, pad_cols)) if pad_cols > 0 else x_flat
            blocks_per_row = int(getattr(group, "stage_ids_0").shape[1])
            x_blocks = x_padded.view(x_flat.shape[0], blocks_per_row, int(group.block_size))
            block_energy = x_blocks.float().norm(dim=-1).mean(dim=0)  # [blocks_per_row]
            row_scale = group.row_scale.float().abs().reshape(-1, 1)  # [n_size, 1]
            for stage_idx in range(group.num_stages):
                ids = getattr(group, f"stage_ids_{stage_idx}").to(torch.int64)  # [n_size, blocks_per_row]
                codebook_norm = self._codebook_norms[stage_idx]
                influence = (
                    row_scale
                    * block_energy.unsqueeze(0)
                    * codebook_norm.index_select(0, ids.reshape(-1)).view_as(ids)
                )
                self.stage_influence[stage_idx].scatter_add_(
                    0, ids.reshape(-1), influence.reshape(-1).to(torch.float64)
                )
                self.stage_counts[stage_idx].scatter_add_(
                    0, ids.reshape(-1), torch.ones(ids.numel(), device=ids.device, dtype=torch.float64)
                )


@torch.no_grad()
def _structural_interference(
    module: PackedGroupedBlockRVQLinear, tile_size: int
) -> tuple[list[torch.Tensor], int, int, int]:
    """Per-stage co-occurrence matrix over tiles of ``tile_size`` consecutive output rows.

    Entry (i, j) = (#tiles containing BOTH id i and id j) / num_tiles.
    Diagonal = (#tiles containing id i) / num_tiles (the marginal).
    """
    interference: list[torch.Tensor | None] = []
    n_size_total = 0
    blocks_per_row_total = 0
    num_tiles_total = 0
    for group in module.groups:
        n_size = int(getattr(group, "stage_ids_0").shape[0])
        blocks_per_row = int(getattr(group, "stage_ids_0").shape[1])
        n_size_total = n_size
        blocks_per_row_total = blocks_per_row
        num_tiles = (n_size + tile_size - 1) // tile_size
        num_tiles_total = num_tiles
        for stage_idx in range(group.num_stages):
            ids = getattr(group, f"stage_ids_{stage_idx}")  # [n_size, blocks_per_row]
            cb_size = int(getattr(group, f"codebook_{stage_idx}").shape[0])
            # Build per-tile presence mask: [num_tiles, M] (bool/uint8).
            presence = torch.zeros((num_tiles, cb_size), dtype=torch.float32, device=ids.device)
            for t in range(num_tiles):
                row_a = t * tile_size
                row_b = min(row_a + tile_size, n_size)
                tile_ids = ids[row_a:row_b].to(torch.int64).reshape(-1)
                # unique-flatten via scatter
                presence[t].index_fill_(0, tile_ids, 1.0)
            # Co-occurrence matrix: M_T @ M = [M, M]
            mat = presence.t() @ presence
            mat /= float(num_tiles)
            while len(interference) <= stage_idx:
                interference.append(None)
            interference[stage_idx] = mat.cpu()
        break  # one group only for these layers
    assert all(m is not None for m in interference), "missing stage interference"
    return [m for m in interference if m is not None], n_size_total, blocks_per_row_total, num_tiles_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="layer names to replace + collect (default: l54 mlp.gate/up)",
    )
    ap.add_argument("--num-windows", type=int, default=2)
    ap.add_argument("--text-source", choices=("raw", "local"), default="raw")
    ap.add_argument("--group-rows", type=int, default=28672)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    ap.add_argument("--tile-size", type=int, default=256, help="rows per Triton tile (= B2 block_n)")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/m23_influence/l54_gate_up.pt"))
    ap.add_argument("--out-summary", default=str(REPO_ROOT / "results/m23_influence/l54_gate_up_summary.json"))
    args = ap.parse_args()

    targets = tuple(args.targets)
    print(f"[targets] {targets}")

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
    layer_stats = replace_with_packed_block_rvq(
        model,
        targets,
        group_rows=args.group_rows,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        num_stages=args.num_stages,
        product_splits=args.product_splits,
        calibrate_stage_scales=True,
        matmul_strategy="full_weight_fast",
    )
    print(f"[encode] {len(layer_stats)} layers in {time.time() - t0:.1f}s", flush=True)

    hooks = []
    trackers: dict[str, _InfluenceHook] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, PackedGroupedBlockRVQLinear) and name in targets:
            trackers[name] = _InfluenceHook(mod)

            def _make(layer_name: str):
                def _hk(_mod, args_in):
                    trackers[layer_name].update(args_in[0].detach())

                return _hk

            hooks.append(mod.register_forward_pre_hook(_make(name)))

    device = model.get_input_embeddings().weight.device
    total_len = ids.size(1)
    num_windows = min(args.num_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    print(f"[calibrate] windows={num_windows}", flush=True)
    with torch.no_grad():
        for i in range(num_windows):
            begin = i * STRIDE
            end = min(begin + MAX_LEN, total_len)
            chunk = ids[:, begin:end].to(device)
            model(chunk)
            print(f"  {i + 1}/{num_windows}", flush=True)
    for h in hooks:
        h.remove()

    print("[interference] computing structural co-occurrence ...", flush=True)
    payload: dict = {"args": vars(args)}
    summary_rows: list[dict] = []
    for name in targets:
        mod = trackers[name].module
        inter, n_size, bpr, num_tiles = _structural_interference(mod, args.tile_size)
        cb_size = int(trackers[name].stage_influence[0].shape[0])
        num_stages = len(trackers[name].stage_influence)
        payload[name] = {
            "M": cb_size,
            "num_stages": num_stages,
            "tile_size": int(args.tile_size),
            "num_tiles": int(num_tiles),
            "n_size": int(n_size),
            "blocks_per_row": int(bpr),
            "stage_influence": [t.cpu() for t in trackers[name].stage_influence],
            "stage_counts": [t.cpu() for t in trackers[name].stage_counts],
            "stage_interference": inter,
        }
        # quick per-stage summary print
        for s, (inf, cnt, intf) in enumerate(
            zip(payload[name]["stage_influence"], payload[name]["stage_counts"], inter)
        ):
            mass = float(inf.sum())
            top64 = float(torch.topk(inf, k=min(64, inf.numel())).values.sum() / max(mass, 1e-12))
            occupancy = float(intf.diag().mean())  # avg fraction of tiles using a given id
            summary_rows.append(
                {
                    "layer": name,
                    "stage": s,
                    "M": cb_size,
                    "total_influence": mass,
                    "top64_influence_share": top64,
                    "avg_id_tile_occupancy": occupancy,
                }
            )
            print(
                f"  {name:55s} stage={s} top64={top64:.3f} avg_tile_occupancy={occupancy:.3f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[save] {out}")

    summary_path = Path(args.out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "rows": summary_rows,
                "layer_stats": layer_stats,
            },
            indent=2,
        )
    )
    print(f"[save] {summary_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
