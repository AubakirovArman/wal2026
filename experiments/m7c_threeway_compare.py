"""M7: Three-way compare — baseline vs eager-bf16 vs cached-packed.

Runs WikiText-2 raw PPL + throughput + VRAM for:
  1. baseline bf16 (vanilla nn.Linear)
  2. eager-bf16 (route-decode once, then nn.Linear-equivalent)
  3. cached-packed deployment (current default)

Shows the speed/VRAM ceiling proof and which runtime is Pareto-optimal.
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

from dwl2_dynamic_route.src.runtime import (
    replace_with_deployment_runtime,
    replace_with_eager_bf16,
    replace_with_eager_fp8,
    replace_with_eager_hybrid,
)

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
MAX_LEN = 2048
STRIDE = 512


def _reset_peaks():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _vram():
    return [
        {"device": i,
         "allocated_mb": round(torch.cuda.memory_allocated(i) / 2**20, 1),
         "reserved_mb": round(torch.cuda.memory_reserved(i) / 2**20, 1),
         "peak_mb": round(torch.cuda.max_memory_allocated(i) / 2**20, 1)}
        for i in range(torch.cuda.device_count())
    ]


def _peak(snap):
    return max((float(x["peak_mb"]) for x in snap), default=0.0)


def _eval_ids(tok, source):
    if source == "raw":
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tok(text, return_tensors="pt").input_ids.cpu()


def _evaluate(model, ids, n_windows):
    model.eval()
    device = model.get_input_embeddings().weight.device
    nlls, tot, prev = [], 0, 0
    L = ids.size(1)
    n_windows = min(n_windows, max(1, (L - MAX_LEN) // STRIDE + 1))
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_windows):
            b, e = i * STRIDE, min(i * STRIDE + MAX_LEN, L)
            tl = e - prev if i > 0 else e - b
            chunk = ids[:, b:e].to(device)
            tgt = chunk.clone()
            if i > 0:
                tgt[:, :-tl] = -100
            loss = model(chunk, labels=tgt).loss.item()
            nlls.append(loss * tl)
            tot += tl
            prev = e
            if (i + 1) % 8 == 0 or i + 1 == n_windows:
                ppl = math.exp(sum(nlls) / tot)
                dt = time.time() - t0
                print(f"  {i+1}/{n_windows}  ppl={ppl:.4f}  tok/s={tot/max(dt,1e-9):.1f}", flush=True)
    dt = time.time() - t0
    return {"perplexity": math.exp(sum(nlls) / tot),
            "total_tokens": tot, "num_windows": n_windows,
            "elapsed_s": dt, "tok_s": tot / max(dt, 1e-9)}


def _load():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True)


def run_pass(name, replace_fn, eval_ids, n_windows):
    print(f"\n=== {name} ===", flush=True)
    model = _load()
    _reset_peaks()
    layer_stats = replace_fn(model) if replace_fn is not None else []
    replace_vram = _vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _reset_peaks()
    metrics = _evaluate(model, eval_ids, n_windows)
    run_vram = _vram()
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
        "layer_stats_count": len(layer_stats),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-windows", type=int, default=16)
    p.add_argument("--text-source", choices=("raw", "local"), default="raw")
    p.add_argument("--cache-max-mb", type=int, default=128)
    p.add_argument("--sample-limit", type=int, default=2_000_000)
    p.add_argument("--shape-policy-json",
                   default=str(ROOT / "dwl2_dynamic_route/results/m6s_shape_runtime_policy.json"))
    p.add_argument("--eager-shape-policy-json", default=None,
                   help="Optional shape policy for eager-bf16 compare. Default: disabled so eager measures the pure materialized ceiling.")
    p.add_argument("--fp8-shape-policy-json", default=None,
                   help="Optional shape policy for eager-fp8 compare. Default: disabled so fp8 measures the pure materialized ceiling.")
    p.add_argument("--modes", default="baseline,eager,cached",
                   help="comma list: baseline,eager,fp8,cached")
    p.add_argument("--out",
                   default=str(ROOT / "dwl2_dynamic_route/results/m7c_threeway_compare.json"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = _eval_ids(tok, args.text_source)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    result = {"text_source": args.text_source, "num_windows": args.num_windows,
              "cache_max_mb": args.cache_max_mb, "modes": modes}

    if "baseline" in modes:
        result["baseline"] = run_pass("baseline", None, ids, args.num_windows)
    if "eager" in modes:
        result["eager"] = run_pass(
            "eager-bf16",
            lambda m: replace_with_eager_bf16(
                m, sample_limit=args.sample_limit,
                shape_policy_json=args.eager_shape_policy_json),
            ids, args.num_windows)
    if "fp8" in modes:
        result["fp8"] = run_pass(
            "eager-fp8",
            lambda m: replace_with_eager_fp8(
                m, sample_limit=args.sample_limit,
                shape_policy_json=args.fp8_shape_policy_json),
            ids, args.num_windows)
    if "hybrid" in modes:
        result["hybrid"] = run_pass(
            "eager-hybrid",
            lambda m: replace_with_eager_hybrid(
                m, sample_limit=args.sample_limit),
            ids, args.num_windows)
    if "cached" in modes:
        result["cached"] = run_pass(
            "cached-packed",
            lambda m: replace_with_deployment_runtime(
                m, sample_limit=args.sample_limit,
                shape_policy_json=args.shape_policy_json,
                cache_max_mb=args.cache_max_mb),
            ids, args.num_windows)

    # Summary
    if "baseline" in result and "eager" in result:
        result["eager_vs_baseline"] = {
            "ppl_gap": result["eager"]["metrics"]["perplexity"] - result["baseline"]["metrics"]["perplexity"],
            "tok_s_ratio": result["eager"]["metrics"]["tok_s"] / max(result["baseline"]["metrics"]["tok_s"], 1e-9),
            "peak_vram_delta_mb": result["eager"]["eval_peak_mb"] - result["baseline"]["eval_peak_mb"],
        }
    if "baseline" in result and "cached" in result:
        result["cached_vs_baseline"] = {
            "ppl_gap": result["cached"]["metrics"]["perplexity"] - result["baseline"]["metrics"]["perplexity"],
            "tok_s_ratio": result["cached"]["metrics"]["tok_s"] / max(result["baseline"]["metrics"]["tok_s"], 1e-9),
            "peak_vram_delta_mb": result["cached"]["eval_peak_mb"] - result["baseline"]["eval_peak_mb"],
        }
    if "baseline" in result and "fp8" in result:
        result["fp8_vs_baseline"] = {
            "ppl_gap": result["fp8"]["metrics"]["perplexity"] - result["baseline"]["metrics"]["perplexity"],
            "tok_s_ratio": result["fp8"]["metrics"]["tok_s"] / max(result["baseline"]["metrics"]["tok_s"], 1e-9),
            "peak_vram_delta_mb": result["fp8"]["eval_peak_mb"] - result["baseline"]["eval_peak_mb"],
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===")
    for mode in ("baseline", "eager", "fp8", "hybrid", "cached"):
        if mode in result:
            m = result[mode]["metrics"]
            v = result[mode]["eval_peak_mb"]
            print(f"  {mode:10s}  ppl={m['perplexity']:.4f}  tok/s={m['tok_s']:.2f}  peak_mb={v:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
