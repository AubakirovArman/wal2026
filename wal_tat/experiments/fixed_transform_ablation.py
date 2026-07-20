"""Compare direct and randomized-Hadamard ternary PTQ on one untouched matrix.

This experiment is deliberately checkpoint-neutral: it installs the accepted
frontier, replaces one still-BF16 matrix in memory, evaluates a supplied suite,
and restores the original module.  Development data may rank transform seeds;
an audit holdout must receive only a seed fixed by the development run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    default_model_path,
    install_checkpoint,
    load_model,
    seed_everything,
    set_submodule,
    sha256_file,
)
from wal_tat import FixedTernaryLinear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--target-name", default="model.layers.23.self_attn.q_proj")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rht-seed", type=int, action="append", default=[])
    parser.add_argument("--sequences-per-domain", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument(
        "--suite-role",
        default="development-only; may rank transform seeds",
        help="Provenance label recorded verbatim in the result artifact.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_chunks(model, chunks, device: str, batch_size: int) -> dict:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        if len({int(chunk.numel()) for chunk in selected}) != 1:
            raise ValueError("evaluation chunks must have equal lengths")
        batch = torch.stack(selected).to(device)
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(batch.shape[0] * (batch.shape[1] - 1))
    nll = total_nll / total_tokens
    return {"nll": nll, "ppl": math.exp(min(nll, 50.0)), "tokens": total_tokens}


def evaluate_domains(model, gates, device: str, batch_size: int) -> dict:
    return {
        domain: evaluate_chunks(model, chunks, device, batch_size)
        for domain, chunks in gates.items()
    }


@torch.no_grad()
def candidate_record(layer, reference_weight, model, gates, baseline, args) -> dict:
    set_submodule(model, args.target_name, layer.to(args.device))
    metrics = evaluate_domains(model, gates, args.device, args.batch_size)
    ratios = {
        domain: metrics[domain]["nll"] / baseline[domain]["nll"]
        for domain in baseline
    }
    effective = layer.effective_weight().float()
    reference = reference_weight.float()
    mse = float(F.mse_loss(effective, reference).item())
    relative_mse = mse / float(reference.square().mean().clamp_min(1e-12).item())
    histogram = layer.code_histogram()
    return {
        "transform": layer.transform,
        "transform_seed": layer.transform_seed if layer.transform == "rht" else None,
        "metrics": metrics,
        "ratios_to_frontier": ratios,
        "worst_ratio": max(ratios.values()),
        "mean_ratio": sum(ratios.values()) / len(ratios),
        "weight_mse": mse,
        "relative_weight_mse": relative_mse,
        "code_histogram": {str(key): value for key, value in histogram.items()},
        "zero_fraction": histogram[0] / reference.numel(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.sequences_per_domain < 1:
        raise ValueError("batch size and sequence count must be positive")
    seeds = args.rht_seed or [109, 211, 307, 401]
    if len(set(seeds)) != len(seeds):
        raise ValueError("RHT seeds must be unique")
    checkpoint = args.checkpoint.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    gates = {
        domain: chunks[: args.sequences_per_domain]
        for domain, chunks in suite["gates"].items()
    }
    seed_everything(args.seed)
    started = time.monotonic()
    model = load_model(model_path, args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    install_checkpoint(model, payload, args.device)
    target = model.get_submodule(args.target_name)
    if not isinstance(target, nn.Linear):
        raise TypeError("target must be an untouched BF16 nn.Linear")
    original = target
    reference_weight = target.weight.detach().clone()
    reference_bias = None if target.bias is None else target.bias.detach().clone()
    baseline = evaluate_domains(model, gates, args.device, args.batch_size)

    candidates = []
    identity = FixedTernaryLinear.from_weight(
        reference_weight,
        group_size=args.group_size,
        transform="identity",
        bias=reference_bias,
    )
    candidates.append(
        candidate_record(identity, reference_weight, model, gates, baseline, args)
    )
    del identity
    for transform_seed in seeds:
        layer = FixedTernaryLinear.from_weight(
            reference_weight,
            group_size=args.group_size,
            transform="rht",
            transform_seed=transform_seed,
            bias=reference_bias,
        )
        candidates.append(
            candidate_record(layer, reference_weight, model, gates, baseline, args)
        )
        del layer

    set_submodule(model, args.target_name, original)
    ranked = sorted(candidates, key=lambda item: (item["worst_ratio"], item["mean_ratio"]))
    identity_result = next(item for item in candidates if item["transform"] == "identity")
    rht_ranked = [item for item in ranked if item["transform"] == "rht"]
    best_rht = rht_ranked[0]
    result = {
        "schema": "wal-tat-fixed-transform-ablation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "target_name": args.target_name,
        "target_shape": list(reference_weight.shape),
        "group_size": args.group_size,
        "sequences_per_domain": args.sequences_per_domain,
        "batch_size": args.batch_size,
        "baseline": baseline,
        "candidates": candidates,
        "ranking": [
            {
                "transform": item["transform"],
                "transform_seed": item["transform_seed"],
                "worst_ratio": item["worst_ratio"],
                "mean_ratio": item["mean_ratio"],
                "relative_weight_mse": item["relative_weight_mse"],
            }
            for item in ranked
        ],
        "best": ranked[0],
        "best_rht": best_rht,
        "identity": identity_result,
        "rht_beats_identity": best_rht["worst_ratio"] < identity_result["worst_ratio"],
        "checkpoint_mutated": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
