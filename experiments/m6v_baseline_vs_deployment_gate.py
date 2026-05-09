"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
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

from dwl2_dynamic_route.src.runtime import replace_with_deployment_runtime

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512


def _reset_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _vram_snapshot() -> list[dict[str, float]]:
    return [
        {
            "device": index,
            "allocated_mb": round(torch.cuda.memory_allocated(index) / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(index) / 2**20, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated(index) / 2**20, 1),
        }
        for index in range(torch.cuda.device_count())
    ]


def _max_peak_mb(snapshot: list[dict[str, float]]) -> float:
    return max((float(item["peak_mb"]) for item in snapshot), default=0.0)


def _load_eval_ids(tokenizer: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tokenizer(text, return_tensors="pt").input_ids.cpu()


def _evaluate(model: nn.Module, input_ids: torch.Tensor, num_windows: int) -> dict[str, float]:
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls: list[float] = []
    total_tokens = 0
    prev_end = 0
    seq_len = input_ids.size(1)
    max_windows = max(1, (seq_len - MAX_LEN) // STRIDE + 1)
    num_windows = min(num_windows, max_windows)
    start = time.time()
    with torch.no_grad():
        for idx in range(num_windows):
            begin = idx * STRIDE
            end = min(begin + MAX_LEN, seq_len)
            target_len = end - prev_end if idx > 0 else end - begin
            chunk = input_ids[:, begin:end].to(device)
            target = chunk.clone()
            if idx > 0:
                target[:, :-target_len] = -100
            loss = model(chunk, labels=target).loss.item()
            nlls.append(loss * target_len)
            total_tokens += target_len
            prev_end = end
            if (idx + 1) % 8 == 0 or idx + 1 == num_windows:
                ppl = math.exp(sum(nlls) / total_tokens)
                elapsed = time.time() - start
                tok_s = total_tokens / max(elapsed, 1e-12)
                print(f"  eval {idx + 1}/{num_windows} windows  ppl={ppl:.4f} tok/s={tok_s:.2f}", flush=True)
    elapsed = time.time() - start
    return {
        "perplexity": math.exp(sum(nlls) / total_tokens),
        "total_tokens": total_tokens,
        "num_windows": num_windows,
        "elapsed_s": elapsed,
        "tok_s": total_tokens / max(elapsed, 1e-12),
    }


def _runtime_summary(layer_stats: list[dict[str, object]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in layer_stats:
        key = str(item["runtime_decision"])
        summary[key] = summary.get(key, 0) + 1
    return summary


def _load_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--text-source", choices=("raw", "local"), default="raw")
    parser.add_argument("--cache-max-mb", type=int, default=128)
    parser.add_argument(
        "--shape-policy-json",
        default=str(ROOT / "dwl2_dynamic_route/results/m6s_shape_runtime_policy.json"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "dwl2_dynamic_route/results/m6v_baseline_vs_deployment_gate.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_ids = _load_eval_ids(tokenizer, args.text_source)

    print("[m6v] baseline eval", flush=True)
    baseline_model = _load_model()
    _reset_peaks()
    baseline = _evaluate(baseline_model, eval_ids, args.num_windows)
    baseline_vram = _vram_snapshot()
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[m6v] deployment runtime replace", flush=True)
    deployment_model = _load_model()
    _reset_peaks()
    layer_stats = replace_with_deployment_runtime(
        deployment_model,
        sample_limit=args.sample_limit,
        shape_policy_json=args.shape_policy_json,
        cache_max_mb=args.cache_max_mb,
    )
    replace_vram = _vram_snapshot()

    print("[m6v] deployment eval", flush=True)
    _reset_peaks()
    deployment = _evaluate(deployment_model, eval_ids, args.num_windows)
    deployment_vram = _vram_snapshot()

    runtime_summary = _runtime_summary(layer_stats)
    result = {
        "text_source": args.text_source,
        "num_windows": args.num_windows,
        "cache_max_mb": args.cache_max_mb,
        "baseline": baseline,
        "baseline_vram": baseline_vram,
        "baseline_max_peak_mb": _max_peak_mb(baseline_vram),
        "deployment": deployment,
        "deployment_vram": deployment_vram,
        "deployment_replace_vram": replace_vram,
        "deployment_max_peak_mb": _max_peak_mb(deployment_vram),
        "deployment_replace_max_peak_mb": _max_peak_mb(replace_vram),
        "ppl_gap": deployment["perplexity"] - baseline["perplexity"],
        "tok_s_delta": deployment["tok_s"] - baseline["tok_s"],
        "tok_s_ratio": deployment["tok_s"] / max(baseline["tok_s"], 1e-12),
        "peak_vram_delta_mb": _max_peak_mb(deployment_vram) - _max_peak_mb(baseline_vram),
        "layer_count": len(layer_stats),
        "runtime_summary": runtime_summary,
        "grouped_local_targets": [
            {
                "name": item["name"],
                "group_rows": item["group_rows"],
                "group_cols": item["group_cols"],
            }
            for item in layer_stats
            if item["runtime_decision"] == "local_palette_grouped"
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "baseline_ppl": baseline["perplexity"],
        "deployment_ppl": deployment["perplexity"],
        "baseline_tok_s": baseline["tok_s"],
        "deployment_tok_s": deployment["tok_s"],
        "baseline_max_peak_mb": _max_peak_mb(baseline_vram),
        "deployment_max_peak_mb": _max_peak_mb(deployment_vram),
        "runtime_summary": runtime_summary,
    }, indent=2))
    print(f"[m6v] wrote {out_path}")


if __name__ == "__main__":
    main()