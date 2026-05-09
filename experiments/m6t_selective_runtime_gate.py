"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
from __future__ import annotations

import argparse
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

from dwl2_dynamic_route.src.runtime import AdaptiveFusedIDRouteLinear, CachedPackedIDRouteLinear, FusedIDRouteLinear, PackedIDRouteLinear, replace_packed_id_route_layers, replace_with_deployment_runtime, replace_with_hybrid_runtime

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--unsloth--Llama-3.3-70B-Instruct/snapshots/99cd0d2c829e92a67c844f9144c2509632e5c87f"
TEXT_PATH = ROOT / "bk/wikitext2_test.txt"
TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
MAX_LEN = 2048
STRIDE = 512


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


def _adaptive_runtime_state(model: nn.Module) -> dict[str, int]:
    fallback_layers = 0
    fallback_events = 0
    primary_disabled_layers = 0
    validated_layers = 0
    shadow_mismatch_layers = 0
    shadow_mismatch_events = 0
    for module in model.modules():
        if isinstance(module, AdaptiveFusedIDRouteLinear):
            validated_layers += 1
            if module.fallback_count > 0:
                fallback_layers += 1
                fallback_events += int(module.fallback_count)
            if module.shadow_mismatch_count > 0:
                shadow_mismatch_layers += 1
                shadow_mismatch_events += int(module.shadow_mismatch_count)
            if not module._primary_enabled:
                primary_disabled_layers += 1
    return {
        "validated_layers": validated_layers,
        "fallback_layers": fallback_layers,
        "fallback_events": fallback_events,
        "shadow_mismatch_layers": shadow_mismatch_layers,
        "shadow_mismatch_events": shadow_mismatch_events,
        "primary_disabled_layers": primary_disabled_layers,
    }


def _adaptive_runtime_names(model: nn.Module) -> dict[str, list[str]]:
    fallback_layer_names: list[str] = []
    shadow_mismatch_layer_names: list[str] = []
    primary_enabled_layer_names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveFusedIDRouteLinear):
            if module.fallback_count > 0:
                fallback_layer_names.append(name)
            if module.shadow_mismatch_count > 0:
                shadow_mismatch_layer_names.append(name)
            if module._primary_enabled:
                primary_enabled_layer_names.append(name)
    return {
        "fallback_layer_names": fallback_layer_names,
        "shadow_mismatch_layer_names": shadow_mismatch_layer_names,
        "primary_enabled_layer_names": primary_enabled_layer_names,
    }


def _cached_runtime_state(model: nn.Module) -> dict[str, int]:
    cached_layers = 0
    cache_hit_count = 0
    cache_miss_count = 0
    cache_skip_count = 0
    for module in model.modules():
        if isinstance(module, CachedPackedIDRouteLinear):
            cached_layers += 1
            cache_hit_count += int(module.cache_hit_count)
            cache_miss_count += int(module.cache_miss_count)
            cache_skip_count += int(module.cache_skip_count)
    return {
        "cached_layers": cached_layers,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_skip_count": cache_skip_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--text-source", choices=("raw", "local"), default="raw")
    parser.add_argument("--runtime-mode", choices=("fused", "adaptive", "adaptive-shadow", "packed", "cached-packed", "deployment", "hybrid"), default="fused")
    parser.add_argument("--adaptive-validate-calls", type=int, default=1)
    parser.add_argument("--adaptive-rel-mse-tol", type=float, default=1e-4)
    parser.add_argument("--cache-max-mb", type=int, default=128)
    parser.add_argument(
        "--fused-allowlist-json",
        default=str(ROOT / "dwl2_dynamic_route/results/m6u_fused_promotion_policy.json"),
    )
    parser.add_argument(
        "--shape-policy-json",
        default=str(ROOT / "dwl2_dynamic_route/results/m6s_shape_runtime_policy.json"),
    )
    parser.add_argument("--out", default=str(ROOT / "dwl2_dynamic_route/results/m6t_selective_runtime_gate.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    eval_ids = _load_eval_ids(tokenizer, args.text_source)

    if args.runtime_mode == "fused":
        runtime_cls = FusedIDRouteLinear
        runtime_kwargs: dict[str, object] = {}
    elif args.runtime_mode == "adaptive":
        runtime_cls = AdaptiveFusedIDRouteLinear
        runtime_kwargs = {"validate_calls": args.adaptive_validate_calls}
    elif args.runtime_mode == "adaptive-shadow":
        runtime_cls = AdaptiveFusedIDRouteLinear
        runtime_kwargs = {
            "validate_calls": args.adaptive_validate_calls,
            "shadow_validate": True,
            "shadow_rel_mse_tol": args.adaptive_rel_mse_tol,
        }
    elif args.runtime_mode == "cached-packed":
        runtime_cls = CachedPackedIDRouteLinear
        runtime_kwargs = {"max_cache_bytes": args.cache_max_mb * 2**20}
    elif args.runtime_mode == "deployment":
        runtime_cls = CachedPackedIDRouteLinear
        runtime_kwargs = {"max_cache_bytes": args.cache_max_mb * 2**20}
    elif args.runtime_mode == "hybrid":
        runtime_cls = CachedPackedIDRouteLinear
        runtime_kwargs = {"max_cache_bytes": args.cache_max_mb * 2**20}
    else:
        runtime_cls = PackedIDRouteLinear
        runtime_kwargs = {}

    print(f"[m6t] replacing target linears with runtime_mode={args.runtime_mode}", flush=True)
    if args.runtime_mode == "deployment":
        layer_stats = replace_with_deployment_runtime(
            model,
            target_suffixes=TARGETS,
            sample_limit=args.sample_limit,
            shape_policy_json=args.shape_policy_json,
            cache_max_mb=args.cache_max_mb,
        )
    elif args.runtime_mode == "hybrid":
        layer_stats = replace_with_hybrid_runtime(
            model,
            target_suffixes=TARGETS,
            sample_limit=args.sample_limit,
            shape_policy_json=args.shape_policy_json,
            cache_max_mb=args.cache_max_mb,
            fused_allowlist_json=args.fused_allowlist_json,
        )
    else:
        layer_stats = replace_packed_id_route_layers(
            model,
            target_suffixes=TARGETS,
            sample_limit=args.sample_limit,
            runtime_cls=runtime_cls,
            runtime_kwargs=runtime_kwargs,
            shape_policy_json=args.shape_policy_json,
        )
    summary = _runtime_summary(layer_stats)
    print(f"[m6t] runtime summary: {summary}", flush=True)

    print("[m6t] selective runtime eval", flush=True)
    routed = _evaluate(model, eval_ids, args.num_windows)
    adaptive_state = _adaptive_runtime_state(model)
    adaptive_names = _adaptive_runtime_names(model)
    cached_state = _cached_runtime_state(model)

    result = {
        "text_source": args.text_source,
        "runtime_mode": args.runtime_mode,
        "layer_count": len(layer_stats),
        "runtime_summary": summary,
        "grouped_local_layers": summary.get("local_palette_grouped", 0),
        "fused_global_layers": summary.get("global_id_triton", 0),
        "adaptive_global_layers": summary.get("adaptive_global_id", 0),
        "cached_packed_layers": summary.get("cached_packed", 0),
        "adaptive_state": adaptive_state,
        "adaptive_names": adaptive_names,
        "cached_state": cached_state,
        "routed": routed,
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
    print(json.dumps(result, indent=2))
    print(f"[m6t] wrote {out_path}")


if __name__ == "__main__":
    main()