"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.runtime import replace_with_eager_block_rvq, replace_with_packed_block_rvq


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512

L0_QKV = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
)
L0_Q = ("model.layers.0.self_attn.q_proj",)
L0_K = ("model.layers.0.self_attn.k_proj",)
L0_V = ("model.layers.0.self_attn.v_proj",)
L0_QK = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
)
L0_GU = (
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
)
L0_QK_GU = L0_QK + L0_GU
L0_QKV_GU = L0_QKV + (
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
)
L54_Q = ("model.layers.54.self_attn.q_proj",)
L54_K = ("model.layers.54.self_attn.k_proj",)
L54_V = ("model.layers.54.self_attn.v_proj",)
L54_QK = L54_Q + L54_K
L54_QKV = L54_QK + L54_V
L54_GU = (
    "model.layers.54.mlp.gate_proj",
    "model.layers.54.mlp.up_proj",
)
L54_Q_GU = L54_Q + L54_GU
L54_K_GU = L54_K + L54_GU
L54_QK_GU = L54_QK + L54_GU
L54_QKV_GU = L54_QKV + L54_GU


def _prefix_qk(n: int) -> tuple[str, ...]:
    targets = []
    for layer in range(n):
        targets.extend(
            (
                f"model.layers.{layer}.self_attn.q_proj",
                f"model.layers.{layer}.self_attn.k_proj",
            )
        )
    return tuple(targets)


def _prefix_qk_gu(n: int) -> tuple[str, ...]:
    targets = []
    for layer in range(n):
        targets.extend(
            (
                f"model.layers.{layer}.self_attn.q_proj",
                f"model.layers.{layer}.self_attn.k_proj",
                f"model.layers.{layer}.mlp.gate_proj",
                f"model.layers.{layer}.mlp.up_proj",
            )
        )
    return tuple(targets)


def _reset_peaks() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _vram():
    return [
        {
            "device": i,
            "allocated_mb": round(torch.cuda.memory_allocated(i) / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(i) / 2**20, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated(i) / 2**20, 1),
        }
        for i in range(torch.cuda.device_count())
    ]


def _peak(snap) -> float:
    return max((float(x["peak_mb"]) for x in snap), default=0.0)


def _eval_ids(tok, source: str):
    if source == "raw":
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


def _evaluate(model, ids, n_windows: int):
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls, total_tokens, prev_end = [], 0, 0
    total_len = ids.size(1)
    n_windows = min(n_windows, max(1, (total_len - MAX_LEN) // STRIDE + 1))
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_windows):
            begin = i * STRIDE
            end = min(i * STRIDE + MAX_LEN, total_len)
            target_len = end - prev_end if i > 0 else end - begin
            chunk = ids[:, begin:end].to(device)
            target = chunk.clone()
            if i > 0:
                target[:, :-target_len] = -100
            loss = model(chunk, labels=target).loss.item()
            nlls.append(loss * target_len)
            total_tokens += target_len
            prev_end = end
            if (i + 1) % 8 == 0 or i + 1 == n_windows:
                ppl = math.exp(sum(nlls) / total_tokens)
                dt = time.time() - t0
                print(f"  {i+1}/{n_windows}  ppl={ppl:.4f}  tok/s={total_tokens/max(dt,1e-9):.1f}", flush=True)
    dt = time.time() - t0
    return {
        "perplexity": math.exp(sum(nlls) / total_tokens),
        "total_tokens": total_tokens,
        "num_windows": n_windows,
        "elapsed_s": dt,
        "tok_s": total_tokens / max(dt, 1e-9),
    }


def _load():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def _apply_runtime(
    model,
    target_names: tuple[str, ...],
    rvq_cfg: dict[str, object],
    runtime_kind: str,
    *,
    attn_group_rows: int | None,
    mlp_group_rows: int | None,
    attn_runtime_kind: str | None,
    mlp_runtime_kind: str | None,
    attn_transform_kind: str | None,
    mlp_transform_kind: str | None,
    attn_residual_correction: str | None,
    mlp_residual_correction: str | None,
    packed_matmul_strategy: str,
    packed_matmul_chunk_rows: int | None,
    packed_hot_topk: int | None,
    packed_hot_score_mode: str,
    packed_hot_min_stage_share: float,
    packed_hot_score_threshold_ratio: float,
):
    def _replace_fn(kind: str):
        if kind == "eager":
            return replace_with_eager_block_rvq
        if kind == "packed":
            return replace_with_packed_block_rvq
        raise ValueError(f"unknown runtime_kind: {kind}")

    attn_targets = tuple(name for name in target_names if ".self_attn." in name)
    mlp_targets = tuple(name for name in target_names if ".mlp." in name)
    other_targets = tuple(name for name in target_names if name not in set(attn_targets) and name not in set(mlp_targets))
    layer_stats = []
    attn_kind = attn_runtime_kind or runtime_kind
    mlp_kind = mlp_runtime_kind or runtime_kind

    if attn_targets:
        attn_cfg = dict(rvq_cfg)
        if attn_group_rows is not None:
            attn_cfg["group_rows"] = attn_group_rows
        if attn_transform_kind is not None:
            attn_cfg["transform_kind"] = attn_transform_kind
        if attn_residual_correction is not None:
            attn_cfg["residual_correction"] = attn_residual_correction
        if attn_kind == "packed":
            attn_cfg["matmul_strategy"] = packed_matmul_strategy
            attn_cfg["matmul_chunk_rows"] = packed_matmul_chunk_rows
            attn_cfg["hot_topk"] = packed_hot_topk
            attn_cfg["hot_score_mode"] = packed_hot_score_mode
            attn_cfg["hot_min_stage_share"] = packed_hot_min_stage_share
            attn_cfg["hot_score_threshold_ratio"] = packed_hot_score_threshold_ratio
        layer_stats.extend(_replace_fn(attn_kind)(model, attn_targets, **attn_cfg))
    if mlp_targets:
        mlp_cfg = dict(rvq_cfg)
        if mlp_group_rows is not None:
            mlp_cfg["group_rows"] = mlp_group_rows
        if mlp_transform_kind is not None:
            mlp_cfg["transform_kind"] = mlp_transform_kind
        if mlp_residual_correction is not None:
            mlp_cfg["residual_correction"] = mlp_residual_correction
        if mlp_kind == "packed":
            mlp_cfg["matmul_strategy"] = packed_matmul_strategy
            mlp_cfg["matmul_chunk_rows"] = packed_matmul_chunk_rows
            mlp_cfg["hot_topk"] = packed_hot_topk
            mlp_cfg["hot_score_mode"] = packed_hot_score_mode
            mlp_cfg["hot_min_stage_share"] = packed_hot_min_stage_share
            mlp_cfg["hot_score_threshold_ratio"] = packed_hot_score_threshold_ratio
        layer_stats.extend(_replace_fn(mlp_kind)(model, mlp_targets, **mlp_cfg))
    if other_targets:
        other_cfg = dict(rvq_cfg)
        if runtime_kind == "packed":
            other_cfg["matmul_strategy"] = packed_matmul_strategy
            other_cfg["matmul_chunk_rows"] = packed_matmul_chunk_rows
            other_cfg["hot_topk"] = packed_hot_topk
            other_cfg["hot_score_mode"] = packed_hot_score_mode
            other_cfg["hot_min_stage_share"] = packed_hot_min_stage_share
            other_cfg["hot_score_threshold_ratio"] = packed_hot_score_threshold_ratio
        layer_stats.extend(_replace_fn(runtime_kind)(model, other_targets, **other_cfg))
    return layer_stats


def run_pass(
    name: str,
    target_names: tuple[str, ...],
    eval_ids,
    num_windows: int,
    rvq_cfg: dict[str, object],
    runtime_kind: str,
    *,
    attn_group_rows: int | None,
    mlp_group_rows: int | None,
    attn_runtime_kind: str | None,
    mlp_runtime_kind: str | None,
    attn_transform_kind: str | None,
    mlp_transform_kind: str | None,
    attn_residual_correction: str | None,
    mlp_residual_correction: str | None,
    packed_matmul_strategy: str,
    packed_matmul_chunk_rows: int | None,
    packed_hot_topk: int | None,
    packed_hot_score_mode: str,
    packed_hot_min_stage_share: float,
    packed_hot_score_threshold_ratio: float,
    seed: int | None = None,
    effective_stages_per_split: int | None = None,
    attn_stages_per_split: int | None = None,
    mlp_stages_per_split: int | None = None,
    per_layer_stages_json: str | None = None,
):
    print(f"\n=== {name} ===", flush=True)
    model = _load()
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    _reset_peaks()
    layer_stats = _apply_runtime(
        model,
        target_names,
        rvq_cfg,
        runtime_kind,
        attn_group_rows=attn_group_rows,
        mlp_group_rows=mlp_group_rows,
        attn_runtime_kind=attn_runtime_kind,
        mlp_runtime_kind=mlp_runtime_kind,
        attn_transform_kind=attn_transform_kind,
        mlp_transform_kind=mlp_transform_kind,
        attn_residual_correction=attn_residual_correction,
        mlp_residual_correction=mlp_residual_correction,
        packed_matmul_strategy=packed_matmul_strategy,
        packed_matmul_chunk_rows=packed_matmul_chunk_rows,
        packed_hot_topk=packed_hot_topk,
        packed_hot_score_mode=packed_hot_score_mode,
        packed_hot_min_stage_share=packed_hot_min_stage_share,
        packed_hot_score_threshold_ratio=packed_hot_score_threshold_ratio,
    )
    if effective_stages_per_split is not None:
        from dwl2_dynamic_route.src.runtime import set_global_effective_stages
        n_groups = set_global_effective_stages(model, int(effective_stages_per_split))
        print(f"  [M21] set effective_stages_per_split={effective_stages_per_split} on {n_groups} groups", flush=True)
    if attn_stages_per_split is not None or mlp_stages_per_split is not None:
        from dwl2_dynamic_route.src.runtime import set_effective_stages_by_name
        counts = set_effective_stages_by_name(model, attn_stages_per_split, mlp_stages_per_split)
        print(f"  [M22] attn_stages={attn_stages_per_split} mlp_stages={mlp_stages_per_split} layers={counts}", flush=True)
    if per_layer_stages_json is not None:
        from dwl2_dynamic_route.src.runtime import set_effective_stages_from_map
        cfg = json.loads(Path(per_layer_stages_json).read_text())
        name_to_k = {r["name"]: int(r["chosen_k"]) for r in cfg.get("rows", [])}
        updated = set_effective_stages_from_map(model, name_to_k)
        from collections import Counter
        dist = Counter(name_to_k.values())
        print(f"  [M24] per-layer stages applied to {updated} layers; chosen_k distribution: {dict(dist)}", flush=True)
    replace_vram = _vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peaks()
    metrics = _evaluate(model, eval_ids, num_windows)
    run_vram = _vram()
    avg_rel_mse = sum(item["rel_mse"] for item in layer_stats) / max(len(layer_stats), 1)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "eval_vram": run_vram,
        "replace_vram": replace_vram,
        "eval_peak_mb": _peak(run_vram),
        "replace_peak_mb": _peak(replace_vram),
        "layer_stats": layer_stats,
        "avg_layer_rel_mse": avg_rel_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--text-source", choices=("raw", "local"), default="raw")
    parser.add_argument("--modes", default="l0_qkv,l0_qkv_gu")
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m10c_block_rvq_global_eval.json"))
    parser.add_argument("--runtime-kind", choices=("eager", "packed"), default="eager")
    parser.add_argument("--attn-runtime-kind", choices=("eager", "packed"), default=None)
    parser.add_argument("--mlp-runtime-kind", choices=("eager", "packed"), default=None)
    parser.add_argument(
        "--packed-matmul-strategy",
        choices=(
            "per_group",
            "per_group_fast",
            "full_weight",
            "full_weight_fast",
            "full_weight_hot",
            "full_weight_hot_v2",
            "chunked_weight",
            "stacked_matmul",
            "stage_local_hot_cold",
            "stage_local_hot_cold_b1",
            "stage_local_hot_cold_b2",
            "stage_local_hot_cold_b3",
            "triton_block_rvq",
        ),
        default="per_group",
    )
    parser.add_argument("--packed-matmul-chunk-rows", type=int, default=None)
    parser.add_argument("--packed-hot-topk", type=int, default=None)
    parser.add_argument("--packed-hot-score-mode", choices=("count", "row_scale_norm", "stage_influence_proxy"), default="row_scale_norm")
    parser.add_argument("--packed-hot-min-stage-share", type=float, default=0.0)
    parser.add_argument("--packed-hot-score-threshold-ratio", type=float, default=0.0)
    parser.add_argument(
        "--effective-stages-per-split",
        type=int,
        default=None,
        help="M21: globally cap residual stages per split (drop high-order residuals).",
    )
    parser.add_argument("--attn-stages-per-split", type=int, default=None, help="M22: stage cap for self_attn.* layers only.")
    parser.add_argument("--mlp-stages-per-split", type=int, default=None, help="M22: stage cap for mlp.* layers only.")
    parser.add_argument("--per-layer-stages-json", type=str, default=None, help="M24: JSON path with rows=[{name, chosen_k}].")
    parser.add_argument("--group-rows", type=int, default=2048)
    parser.add_argument("--attn-group-rows", type=int, default=None)
    parser.add_argument("--mlp-group-rows", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--product-splits", type=int, default=1)
    parser.add_argument("--normalize-blocks", choices=("none", "amax", "l2"), default="none")
    parser.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default="none")
    parser.add_argument("--attn-transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default=None)
    parser.add_argument("--mlp-transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default=None)
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--residual-correction", choices=("none", "sign"), default="none")
    parser.add_argument("--attn-residual-correction", choices=("none", "sign"), default=None)
    parser.add_argument("--mlp-residual-correction", choices=("none", "sign"), default=None)
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)
    rvq_cfg = {
        "group_rows": args.group_rows,
        "block_size": args.block_size,
        "codebook_size": args.codebook_size,
        "num_stages": args.num_stages,
        "product_splits": args.product_splits,
        "normalize_blocks": args.normalize_blocks,
        "transform_kind": args.transform_kind,
        "calibrate_stage_scales": args.calibrate_stage_scales,
        "residual_correction": args.residual_correction,
        "sample_limit": args.sample_limit,
        "kmeans_iters": args.kmeans_iters,
        "batch_size": args.batch_size,
    }
    mode_map = {
        "l0_q": L0_Q,
        "l0_k": L0_K,
        "l0_v": L0_V,
        "l0_qk": L0_QK,
        "l0_gu": L0_GU,
        "l0_qk_gu": L0_QK_GU,
        "l0_qkv": L0_QKV,
        "l0_qkv_gu": L0_QKV_GU,
        "l54_q": L54_Q,
        "l54_k": L54_K,
        "l54_v": L54_V,
        "l54_qk": L54_QK,
        "l54_qkv": L54_QKV,
        "l54_gu": L54_GU,
        "l54_q_gu": L54_Q_GU,
        "l54_k_gu": L54_K_GU,
        "l54_qk_gu": L54_QK_GU,
        "l54_qkv_gu": L54_QKV_GU,
        "first2_qk": _prefix_qk(2),
        "first4_qk": _prefix_qk(4),
        "first8_qk": _prefix_qk(8),
        "first2_qk_gu": _prefix_qk_gu(2),
        "first4_qk_gu": _prefix_qk_gu(4),
        "first8_qk_gu": _prefix_qk_gu(8),
    }

    results = {
        "text_source": args.text_source,
        "num_windows": args.num_windows,
        "runtime_kind": args.runtime_kind,
        "rvq_cfg": rvq_cfg,
        "modes": [],
    }
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        target_names = mode_map[mode]
        results[mode] = run_pass(
            mode,
            target_names,
            ids,
            args.num_windows,
            rvq_cfg,
            args.runtime_kind,
            attn_group_rows=args.attn_group_rows,
            mlp_group_rows=args.mlp_group_rows,
            attn_runtime_kind=args.attn_runtime_kind,
            mlp_runtime_kind=args.mlp_runtime_kind,
            attn_transform_kind=args.attn_transform_kind,
            mlp_transform_kind=args.mlp_transform_kind,
            attn_residual_correction=args.attn_residual_correction,
            mlp_residual_correction=args.mlp_residual_correction,
            packed_matmul_strategy=args.packed_matmul_strategy,
            packed_matmul_chunk_rows=args.packed_matmul_chunk_rows,
            packed_hot_topk=args.packed_hot_topk,
            packed_hot_score_mode=args.packed_hot_score_mode,
            packed_hot_min_stage_share=args.packed_hot_min_stage_share,
            packed_hot_score_threshold_ratio=args.packed_hot_score_threshold_ratio,
            seed=args.seed,
            effective_stages_per_split=args.effective_stages_per_split,
            attn_stages_per_split=args.attn_stages_per_split,
            mlp_stages_per_split=args.mlp_stages_per_split,
            per_layer_stages_json=args.per_layer_stages_json,
        )
        results["modes"].append(mode)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY ===")
    for mode in results["modes"]:
        metrics = results[mode]["metrics"]
        print(
            f"  {mode:12s} ppl={metrics['perplexity']:.4f} tok/s={metrics['tok_s']:.2f} "
            f"peak_mb={results[mode]['eval_peak_mb']:.1f} avg_layer_rel_mse={results[mode]['avg_layer_rel_mse']:.3e}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()