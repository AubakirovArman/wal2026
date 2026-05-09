"""M24: per-layer calibrated stage cap.

Idea: at calibration time, push N batches of real activations through each
PackedGroupedBlockRVQLinear at full stages (k=stages_per_split max) and at
candidate reduced stages. For each layer pick the smallest k whose output
cosine similarity stays above a threshold. Saves to JSON for use at eval.

This is the "lingvistic importance" knob — every layer gets its own
'how many phonemes are needed to still be understood' setting.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from dwl2_dynamic_route.src.runtime import (  # noqa: E402
    PackedBlockRVQGroup,
    PackedGroupedBlockRVQLinear,
    replace_with_packed_block_rvq,
    set_global_effective_stages,
)


MODEL_DIR = REPO_ROOT.parent / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"


def _layer_stages(layer: PackedGroupedBlockRVQLinear) -> int:
    return max(int(g.stages_per_split[0]) for g in layer.groups)


def _set_layer_stages(layer: PackedGroupedBlockRVQLinear, k: int) -> None:
    for g in layer.groups:
        g.set_effective_stages_per_split(int(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-windows", type=int, default=4, help="calibration windows")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--targets-mode", default="first8_qk_gu", help="which mode in m10c maps to targets")
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--product-splits", type=int, default=4)
    ap.add_argument("--group-rows", type=int, default=28672)
    ap.add_argument("--cosine-threshold", type=float, default=0.999, help="min cos to keep a layer's output similar to k=max")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/m24_per_layer_stages.json"))
    args = ap.parse_args()

    # Build target list (mirror m10c)
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from m10c_block_rvq_global_eval import (  # noqa
        L54_Q_GU, L54_QK_GU, L54_QKV_GU, _prefix_qk, _prefix_qk_gu,
    )
    mode_map = {
        "l54_q_gu": L54_Q_GU,
        "l54_qk_gu": L54_QK_GU,
        "l54_qkv_gu": L54_QKV_GU,
        "first2_qk_gu": _prefix_qk_gu(2),
        "first4_qk_gu": _prefix_qk_gu(4),
        "first8_qk_gu": _prefix_qk_gu(8),
    }
    targets = mode_map[args.targets_mode]
    print(f"[targets] mode={args.targets_mode} count={len(targets)}")

    print("[load] model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True, low_cpu_mem_usage=True,
    )
    model.eval()

    print("[encode] packing target layers ...")
    rvq_cfg = dict(
        group_rows=args.group_rows, block_size=32, codebook_size=256,
        num_stages=args.num_stages, product_splits=args.product_splits,
        calibrate_stage_scales=True,
    )
    attn_targets = tuple(n for n in targets if ".self_attn." in n)
    mlp_targets = tuple(n for n in targets if ".mlp." in n)
    if attn_targets:
        replace_with_packed_block_rvq(model, attn_targets, **rvq_cfg, matmul_strategy="full_weight_fast")
    if mlp_targets:
        replace_with_packed_block_rvq(model, mlp_targets, **rvq_cfg, matmul_strategy="full_weight_fast")

    # Collect packed layers by name
    packed_layers: Dict[str, PackedGroupedBlockRVQLinear] = {}
    for name, m in model.named_modules():
        if isinstance(m, PackedGroupedBlockRVQLinear):
            packed_layers[name] = m
    print(f"[packed] {len(packed_layers)} layers")

    # Hook each packed layer to capture (input, output_at_max_k) over a few batches
    captures: Dict[str, List[torch.Tensor]] = {n: [] for n in packed_layers}
    inputs:   Dict[str, List[torch.Tensor]] = {n: [] for n in packed_layers}

    set_global_effective_stages(model, args.num_stages)

    handles = []
    def _make_hook(name):
        def hook(mod, args_in, out):
            x = args_in[0]
            inputs[name].append(x.detach())
            captures[name].append(out.detach())
        return hook
    for n, m in packed_layers.items():
        handles.append(m.register_forward_hook(_make_hook(n)))

    print(f"[calibrate] forward {args.num_windows} windows ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for i in range(args.num_windows):
            begin = i * args.max_len
            chunk = ids[:, begin:begin + args.max_len].to(device)
            if chunk.size(1) < args.max_len // 2:
                break
            model(chunk)
    for h in handles:
        h.remove()

    # Now sweep k for each layer; compare output cos to k=max
    rows = []
    for name, layer in packed_layers.items():
        max_k = _layer_stages(layer)
        ref_outs = captures[name]
        ins = inputs[name]
        chosen = max_k
        per_k_cos = {}
        for k in range(max_k - 1, 0, -1):
            _set_layer_stages(layer, k)
            cos_vals = []
            with torch.no_grad():
                for x_in, ref in zip(ins, ref_outs):
                    out = layer(x_in)
                    a = out.float().reshape(-1)
                    b = ref.float().reshape(-1)
                    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                    cos_vals.append(cos)
            mean_cos = float(sum(cos_vals) / max(len(cos_vals), 1))
            per_k_cos[k] = mean_cos
            if mean_cos >= args.cosine_threshold:
                chosen = k
        # restore to max
        _set_layer_stages(layer, max_k)
        rows.append({"name": name, "max_k": max_k, "chosen_k": chosen, "per_k_cos": per_k_cos})
        print(f"  {name:55s}  chosen_k={chosen}/{max_k}  cos@k-1={per_k_cos.get(max_k-1, 'NA'):.5f}" if isinstance(per_k_cos.get(max_k-1), float) else f"  {name}  chosen_k={chosen}/{max_k}")

    out = {"args": vars(args), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[save] {args.out}")
    summary = {}
    for r in rows:
        summary[r["chosen_k"]] = summary.get(r["chosen_k"], 0) + 1
    print(f"[summary] chosen_k distribution: {summary}")


if __name__ == "__main__":
    main()
