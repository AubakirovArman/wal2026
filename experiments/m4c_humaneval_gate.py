from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

import sys
sys.path.insert(0, str(ROOT))
from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.route_encoder import decode_routes, encode_routes, rel_mse


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


def _iter_target_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(TARGETS):
            yield name, module


@torch.no_grad()
def _quantize_model_in_place(model: nn.Module, l_max: int, sample_limit: int) -> list[dict[str, float]]:
    stats = []
    for name, module in _iter_target_linears(model):
        weight = module.weight.detach()
        row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = weight / row_scale
        sample = w_norm.flatten()
        if sample.numel() > sample_limit:
            idx = torch.randint(0, sample.numel(), (sample_limit,), device=sample.device)
            sample = sample[idx]
        ladder = calibrate_ladder(sample, l_max=l_max, refine_iters=20, pin_top=True, top_value=1.0, seed="geometric")
        enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
        w_hat = decode_routes(enc, ladder, out_dtype=weight.dtype) * row_scale.to(weight.dtype)
        module.weight.data.copy_(w_hat)
        stats.append({"name": name, "rel_mse": float(rel_mse(weight.float(), w_hat.float()).item())})
    return stats


def _extract_pass_at_1(task_result: dict) -> float | None:
    for key, value in task_result.items():
        if key.startswith("pass@1"):
            return float(value)
    return None


def _run_humaneval(model: nn.Module, tokenizer, batch_size: int, limit: int | None) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device="cuda",
        dtype="bfloat16",
    )
    start = time.time()
    result = simple_evaluate(
        model=lm,
        tasks=["humaneval"],
        confirm_run_unsafe_code=True,
        num_fewshot=0,
        batch_size=batch_size,
        device="cuda",
        limit=limit,
    )
    task_result = result.get("results", {}).get("humaneval", {})
    return {
        "task_result": task_result,
        "pass@1": _extract_pass_at_1(task_result),
        "elapsed_s": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--l-max", type=int, default=12)
    parser.add_argument("--sample-limit", type=int, default=2_000_000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    print("[m4c] dense HumanEval", flush=True)
    _reset_peaks()
    dense = _run_humaneval(model, tokenizer, args.batch_size, args.limit)
    dense_vram = _vram_snapshot()

    print("[m4c] route surgery", flush=True)
    _reset_peaks()
    layer_stats = _quantize_model_in_place(model, args.l_max, args.sample_limit)
    quant_vram = _vram_snapshot()

    print("[m4c] routed HumanEval", flush=True)
    _reset_peaks()
    routed = _run_humaneval(model, tokenizer, args.batch_size, args.limit)
    routed_vram = _vram_snapshot()

    result = {
        "limit": args.limit,
        "dense": dense,
        "routed": routed,
        "pass@1_gap": None if dense["pass@1"] is None or routed["pass@1"] is None else routed["pass@1"] - dense["pass@1"],
        "dense_vram": dense_vram,
        "quant_vram": quant_vram,
        "routed_vram": routed_vram,
        "layer_count": len(layer_stats),
        "max_rel_mse": max(item["rel_mse"] for item in layer_stats),
    }
    out = ROOT / "dwl2_dynamic_route/results/m4c_humaneval_gate.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[m4c] wrote {out}")


if __name__ == "__main__":
    main()