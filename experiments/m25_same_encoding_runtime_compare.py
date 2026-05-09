from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.block_vq import encode_grouped_block_residual_vq
from dwl2_dynamic_route.src.encoding_io import load_grouped_encoding_map, save_grouped_encoding_map
from dwl2_dynamic_route.src.runtime import replace_with_preencoded_packed_block_rvq


MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512

L0_QKV_GU = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
)
L54_Q_GU = (
    "model.layers.54.self_attn.q_proj",
    "model.layers.54.mlp.gate_proj",
    "model.layers.54.mlp.up_proj",
)


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


MODE_MAP = {
    "l0_qkv_gu": L0_QKV_GU,
    "l54_q_gu": L54_Q_GU,
    "first2_qk_gu": _prefix_qk_gu(2),
    "first8_qk_gu": _prefix_qk_gu(8),
}


def _reset_peaks() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _vram() -> list[dict[str, float]]:
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
    return max((float(item["peak_mb"]) for item in snap), default=0.0)


def _eval_ids(tok, source: str):
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


def _load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


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


def _build_or_load_cache(args, target_names: tuple[str, ...], cache_path: Path) -> dict[str, object]:
    if cache_path.exists() and not args.rebuild_cache:
        print(f"[cache] load {cache_path}", flush=True)
        return load_grouped_encoding_map(cache_path)

    print(f"[cache] build {cache_path}", flush=True)
    model = _load_model()
    target_set = set(target_names)
    encodings = {}
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        print(f"  encode {name}", flush=True)
        encodings[name] = encode_grouped_block_residual_vq(
            module.weight.detach(),
            group_rows=args.group_rows,
            block_size=args.block_size,
            codebook_size=args.codebook_size,
            num_stages=args.num_stages,
            product_splits=args.product_splits,
            normalize_blocks=args.normalize_blocks,
            transform_kind=args.transform_kind,
            calibrate_stage_scales=args.calibrate_stage_scales,
            residual_correction=args.residual_correction,
            sample_limit=args.sample_limit,
            kmeans_iters=args.kmeans_iters,
            batch_size=args.batch_size,
        )
    save_grouped_encoding_map(cache_path, encodings)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encodings


def _run_strategy(args, ids, cache_path: Path, strategy: str):
    print(f"\n=== {strategy} ===", flush=True)
    model = _load_model()
    encodings = load_grouped_encoding_map(cache_path)
    _reset_peaks()
    layer_stats = replace_with_preencoded_packed_block_rvq(
        model,
        encodings,
        matmul_strategy=strategy,
        matmul_chunk_rows=args.packed_matmul_chunk_rows,
        local_palette_group_cols=args.packed_local_palette_group_cols,
        hot_topk=args.packed_hot_topk,
        hot_score_mode=args.packed_hot_score_mode,
        hot_min_stage_share=args.packed_hot_min_stage_share,
        hot_score_threshold_ratio=args.packed_hot_score_threshold_ratio,
    )
    replace_vram = _vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peaks()
    metrics = _evaluate(model, ids, args.num_windows)
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
        "avg_layer_rel_mse": avg_rel_mse,
        "layer_stats": layer_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_MAP), default="l0_qkv_gu")
    parser.add_argument("--strategies", default="full_weight_fast,triton_block_rvq,full_weight_hot")
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--text-source", choices=("raw", "local"), default="raw")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--group-rows", type=int, default=28672)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--product-splits", type=int, default=4)
    parser.add_argument("--normalize-blocks", choices=("none", "amax", "l2"), default="none")
    parser.add_argument("--transform-kind", choices=("none", "dct", "hadamard", "rand_hadamard", "polar", "pca"), default="none")
    parser.add_argument("--calibrate-stage-scales", action="store_true")
    parser.add_argument("--residual-correction", choices=("none", "sign"), default="none")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--packed-matmul-chunk-rows", type=int, default=None)
    parser.add_argument("--packed-local-palette-group-cols", type=int, default=256)
    parser.add_argument("--packed-hot-topk", type=int, default=16)
    parser.add_argument("--packed-hot-score-mode", choices=("count", "row_scale_norm", "stage_influence_proxy"), default="row_scale_norm")
    parser.add_argument("--packed-hot-min-stage-share", type=float, default=0.6)
    parser.add_argument("--packed-hot-score-threshold-ratio", type=float, default=0.0)
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m25_same_encoding_runtime_compare.json"))
    args = parser.parse_args()

    cache_path = Path(args.cache_path) if args.cache_path is not None else ROOT / f"dwl2_dynamic_route/results/m25_{args.mode}_encodings.pt"
    target_names = MODE_MAP[args.mode]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    _build_or_load_cache(args, target_names, cache_path)

    result = {
        "mode": args.mode,
        "text_source": args.text_source,
        "num_windows": args.num_windows,
        "cache_path": str(cache_path),
        "strategies": strategies,
    }
    for strategy in strategies:
        result[strategy] = _run_strategy(args, ids, cache_path, strategy)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===")
    for strategy in strategies:
        metrics = result[strategy]["metrics"]
        print(
            f"  {strategy:20s} ppl={metrics['perplexity']:.4f} tok/s={metrics['tok_s']:.2f} "
            f"peak_mb={result[strategy]['eval_peak_mb']:.1f} avg_layer_rel_mse={result[strategy]['avg_layer_rel_mse']:.3e}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()