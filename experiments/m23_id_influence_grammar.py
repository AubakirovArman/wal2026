"""M23: activation-weighted ID influence grammar for packed Block-RVQ layers.

Builds a per-layer "grammar" over discrete weight tokens. For every packed
Block-RVQ layer we accumulate how much each `(stage, id)` contributes under
real activations from calibration windows.

Influence model:
    influence(row, block, stage) ~= ||x_block||_2 * row_scale[row] * ||codebook[id]||_2

This is not a downstream quality metric; it is a routing prior that answers:
which discrete IDs are the hot words of a layer's language?

Output:
  - per-layer concentration metrics: top8/top16/top32/top64 influence share
  - effective weighted vocab size (exp(entropy))
  - top tokens `(stage,id)` by influence share
  - per-stage histograms for later hot/cold split experiments
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from dwl2_dynamic_route.src.runtime import PackedBlockRVQGroup, PackedGroupedBlockRVQLinear, replace_with_packed_block_rvq  # noqa: E402

MODEL_DIR = WORKSPACE_ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TEXT_PATH = WORKSPACE_ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512


L54_Q_GU = (
    "model.language_model.layers.54.self_attn.q_proj",
    "model.language_model.layers.54.mlp.gate_proj",
    "model.language_model.layers.54.mlp.up_proj",
)
L54_QK_GU = (
    "model.language_model.layers.54.self_attn.q_proj",
    "model.language_model.layers.54.self_attn.k_proj",
    "model.language_model.layers.54.mlp.gate_proj",
    "model.language_model.layers.54.mlp.up_proj",
)


def _prefix_qk_gu(n: int) -> tuple[str, ...]:
    targets = []
    for layer in range(n):
        targets.extend(
            (
                f"model.language_model.layers.{layer}.self_attn.q_proj",
                f"model.language_model.layers.{layer}.self_attn.k_proj",
                f"model.language_model.layers.{layer}.mlp.gate_proj",
                f"model.language_model.layers.{layer}.mlp.up_proj",
            )
        )
    return tuple(targets)


MODE_MAP = {
    "l54_q_gu": L54_Q_GU,
    "l54_qk_gu": L54_QK_GU,
    "first2_qk_gu": _prefix_qk_gu(2),
    "first4_qk_gu": _prefix_qk_gu(4),
    "first8_qk_gu": _prefix_qk_gu(8),
}


def _eval_ids(tok: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


class IDInfluenceTracker:
    def __init__(self, module: PackedGroupedBlockRVQLinear) -> None:
        self.module = module
        self.stage_influence: list[torch.Tensor] = []
        self.stage_counts: list[torch.Tensor] = []
        self.stage_codebook_norms: list[torch.Tensor] = []
        self.stage_total_occurrences = 0
        # Prebuild codebook norms merged with stage scales.
        for group in module.groups:
            for stage_idx in range(group.num_stages):
                while len(self.stage_influence) <= stage_idx:
                    codebook = getattr(group, f"codebook_{stage_idx}")
                    self.stage_influence.append(torch.zeros(int(codebook.shape[0]), dtype=torch.float64, device=codebook.device))
                    self.stage_counts.append(torch.zeros(int(codebook.shape[0]), dtype=torch.float64, device=codebook.device))
                    self.stage_codebook_norms.append(torch.empty(0, device=codebook.device))
                if self.stage_codebook_norms[stage_idx].numel() == 0:
                    codebook = getattr(group, f"codebook_{stage_idx}").float()
                    if group.stage_scales is not None:
                        codebook = codebook * group.stage_scales[stage_idx].float()
                    self.stage_codebook_norms[stage_idx] = codebook.norm(dim=-1).contiguous()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_flat = x.reshape(-1, x.shape[-1])
        for group in self.module.groups:
            pad_cols = int(group.padded_cols) - int(group.in_features)
            if pad_cols > 0:
                x_padded = F.pad(x_flat, (0, pad_cols))
            else:
                x_padded = x_flat
            blocks_per_row = int(getattr(group, "stage_ids_0").shape[1])
            x_blocks = x_padded.view(x_flat.shape[0], blocks_per_row, int(group.block_size))
            block_energy = x_blocks.float().norm(dim=-1).mean(dim=0)
            row_scale = group.row_scale.float().abs().reshape(-1, 1)
            for stage_idx in range(group.num_stages):
                ids = getattr(group, f"stage_ids_{stage_idx}").to(torch.int64)
                codebook_norm = self.stage_codebook_norms[stage_idx]
                influence = row_scale * block_energy.unsqueeze(0) * codebook_norm.index_select(0, ids.reshape(-1)).view_as(ids)
                self.stage_influence[stage_idx].scatter_add_(0, ids.reshape(-1), influence.reshape(-1).to(torch.float64))
                self.stage_counts[stage_idx].scatter_add_(0, ids.reshape(-1), torch.ones(ids.numel(), device=ids.device, dtype=torch.float64))
                self.stage_total_occurrences += ids.numel()

    def summary(self, top_k: int = 16) -> dict[str, object]:
        stage_rows = []
        merged = []
        total_mass = 0.0
        for stage_idx, (inf_hist, cnt_hist) in enumerate(zip(self.stage_influence, self.stage_counts)):
            mass = inf_hist.sum().item()
            total_mass += mass
            top_vals, top_ids = torch.topk(inf_hist, k=min(top_k, inf_hist.numel()))
            nz = inf_hist[inf_hist > 0]
            probs = nz / max(nz.sum(), 1e-12)
            entropy = float(-(probs * probs.log()).sum().item()) if nz.numel() else 0.0
            stage_rows.append(
                {
                    "stage": stage_idx,
                    "total_influence": mass,
                    "effective_vocab": float(math.exp(entropy)),
                    "counts_total": float(cnt_hist.sum().item()),
                    "top_ids": [
                        {
                            "id": int(i.item()),
                            "influence": float(v.item()),
                            "share": float(v.item() / max(mass, 1e-12)),
                            "count": int(cnt_hist[int(i.item())].item()),
                        }
                        for v, i in zip(top_vals, top_ids)
                        if v.item() > 0.0
                    ],
                }
            )
            for token_id, token_mass in enumerate(inf_hist.tolist()):
                if token_mass > 0.0:
                    merged.append((stage_idx, token_id, float(token_mass), int(cnt_hist[token_id].item())))
        merged.sort(key=lambda item: item[2], reverse=True)
        merged_mass = sum(item[2] for item in merged)
        shares = [item[2] / max(merged_mass, 1e-12) for item in merged]
        entropy = -sum(p * math.log(p) for p in shares if p > 0.0)
        def _top_share(k: int) -> float:
            return float(sum(item[2] for item in merged[:k]) / max(merged_mass, 1e-12))
        return {
            "stage_total_occurrences": int(self.stage_total_occurrences),
            "total_influence": float(total_mass),
            "effective_vocab": float(math.exp(entropy)) if merged else 0.0,
            "top8_share": _top_share(8),
            "top16_share": _top_share(16),
            "top32_share": _top_share(32),
            "top64_share": _top_share(64),
            "top_tokens": [
                {
                    "stage": int(stage_idx),
                    "id": int(token_id),
                    "influence": float(token_mass),
                    "share": float(token_mass / max(merged_mass, 1e-12)),
                    "count": int(count),
                }
                for stage_idx, token_id, token_mass, count in merged[:top_k]
            ],
            "stages": stage_rows,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=tuple(MODE_MAP), default="first8_qk_gu")
    ap.add_argument("--num-windows", type=int, default=2)
    ap.add_argument("--text-source", choices=("raw", "local"), default="raw")
    ap.add_argument("--group-rows", type=int, default=28672)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/m23_id_influence_grammar.json"))
    args = ap.parse_args()

    targets = MODE_MAP[args.mode]
    print(f"[targets] mode={args.mode} count={len(targets)}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    print("[load] model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("[encode] replace target layers with packed Block-RVQ ...", flush=True)
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
    print(f"[packed] layers={len(layer_stats)}", flush=True)

    trackers: dict[str, IDInfluenceTracker] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, PackedGroupedBlockRVQLinear):
            trackers[name] = IDInfluenceTracker(module)
            def _make_hook(layer_name: str):
                def _hook(mod, args_in):
                    trackers[layer_name].update(args_in[0].detach())
                return _hook
            hooks.append(module.register_forward_pre_hook(_make_hook(name)))

    device = model.get_input_embeddings().weight.device
    total_len = ids.size(1)
    num_windows = min(args.num_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    print(f"[calibrate] windows={num_windows}", flush=True)
    with torch.no_grad():
        for i in range(num_windows):
            begin = i * STRIDE
            end = min(i * STRIDE + MAX_LEN, total_len)
            chunk = ids[:, begin:end].to(device)
            model(chunk)
            print(f"  {i + 1}/{num_windows}", flush=True)

    for hook in hooks:
        hook.remove()

    rows = []
    for name, tracker in trackers.items():
        row = {"name": name, **tracker.summary(top_k=args.top_k)}
        rows.append(row)
        print(
            f"  {name:55s} top16={row['top16_share']:.3f} top32={row['top32_share']:.3f} "
            f"top64={row['top64_share']:.3f} eff_vocab={row['effective_vocab']:.1f}",
            flush=True,
        )

    rows.sort(key=lambda item: item["top32_share"], reverse=True)
    output = {
        "args": vars(args),
        "rows": rows,
        "layer_stats": layer_stats,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"[save] {out}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
