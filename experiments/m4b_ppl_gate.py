from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.route_encoder import decode_routes, encode_routes, rel_mse

MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
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


def _reset_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _vram_snapshot() -> list[dict[str, float]]:
    out = []
    for index in range(torch.cuda.device_count()):
        out.append(
            {
                "device": index,
                "allocated_mb": round(torch.cuda.memory_allocated(index) / 2**20, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(index) / 2**20, 1),
                "peak_mb": round(torch.cuda.max_memory_allocated(index) / 2**20, 1),
            }
        )
    return out


def _load_eval_ids(tokenizer: AutoTokenizer, source: str) -> torch.Tensor:
    if source == "raw":
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    else:
        text = TEXT_PATH.read_text()
    return tokenizer(text, return_tensors="pt").input_ids.cpu()


def _evaluate_ppl(model: nn.Module, input_ids: torch.Tensor, num_windows: int) -> dict[str, float]:
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
                print(f"  eval {idx + 1}/{num_windows} windows  ppl={ppl:.4f}", flush=True)
    return {
        "perplexity": math.exp(sum(nlls) / total_tokens),
        "total_tokens": total_tokens,
        "num_windows": num_windows,
        "elapsed_s": round(time.time() - start, 1),
    }


def _iter_target_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(TARGETS):
            yield name, module


@torch.no_grad()
def _quantize_model_in_place(model: nn.Module, l_max: int, sample_limit: int) -> list[dict[str, float]]:
    stats = []
    start = time.time()
    for index, (name, module) in enumerate(_iter_target_linears(model), start=1):
        weight = module.weight.detach()
        row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = weight / row_scale
        sample = w_norm.flatten()
        if sample.numel() > sample_limit:
            choice = torch.randint(0, sample.numel(), (sample_limit,), device=sample.device)
            sample = sample[choice]
        ladder = calibrate_ladder(
            sample,
            l_max=l_max,
            refine_iters=20,
            pin_top=True,
            top_value=1.0,
            seed="geometric",
        )
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
        w_hat = decode_routes(enc, ladder, out_dtype=weight.dtype) * row_scale.to(weight.dtype)
        module.weight.data.copy_(w_hat)
        stats.append(
            {
                "name": name,
                "rel_mse": float(rel_mse(weight.float(), w_hat.float()).item()),
            }
        )
        if index % 20 == 0:
            elapsed = time.time() - start
            print(f"  quantized {index} layers  elapsed={elapsed:.0f}s", flush=True)
            torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--l-max", type=int, default=12)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    parser.add_argument("--text-source", choices=("raw", "local"), default="raw")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    eval_ids = _load_eval_ids(tokenizer, args.text_source)

    print("[m4b] baseline PPL", flush=True)
    _reset_peaks()
    baseline = _evaluate_ppl(model, eval_ids, args.num_windows)
    baseline_vram = _vram_snapshot()

    print("[m4b] route-quantizing block linears in-place", flush=True)
    _reset_peaks()
    layer_stats = _quantize_model_in_place(model, args.l_max, args.sample_limit)
    quant_vram = _vram_snapshot()

    print("[m4b] routed PPL", flush=True)
    _reset_peaks()
    routed = _evaluate_ppl(model, eval_ids, args.num_windows)
    routed_vram = _vram_snapshot()

    result = {
        "baseline": baseline,
        "routed": routed,
        "text_source": args.text_source,
        "ppl_gap": routed["perplexity"] - baseline["perplexity"],
        "baseline_vram": baseline_vram,
        "quant_vram": quant_vram,
        "routed_vram": routed_vram,
        "layer_count": len(layer_stats),
        "max_rel_mse": max(item["rel_mse"] for item in layer_stats),
        "worst_layers": sorted(layer_stats, key=lambda item: item["rel_mse"], reverse=True)[:5],
    }
    out_path = ROOT / "dwl2_dynamic_route/results/m4b_ppl_gate.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[m4b] wrote {out_path}")


if __name__ == "__main__":
    main()