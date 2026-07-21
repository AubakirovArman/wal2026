"""Rank unconverted decoder blocks by a causal ternarization-damage proxy."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wal_tat import CausalMomentCollector, activation_fisher_group_damage

from compensation_window import (
    PROJECT,
    WORKSPACE,
    default_model_path,
    install_checkpoint,
    load_model,
    log,
    seed_everything,
    sha256_file,
)


PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=WORKSPACE
        / "wal2/checkpoints/wal-tat-block27_code_recovery_stage3_c4x4-all-codes.pt",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-diverse-recovery-v1-c4x4-l256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", default="decoder_sensitivity_v1")
    parser.add_argument("--samples-per-domain", type=int, default=4)
    parser.add_argument("--domain-offset", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def balanced_calibration(suite, count: int, offset: int):
    per_domain = int(suite["calibration_per_domain"])
    sizes = (
        per_domain * int(suite.get("c4_calibration_repeat", 1)),
        per_domain * int(suite.get("squad_calibration_repeat", 1)),
        per_domain * int(suite.get("code_calibration_repeat", 1)),
    )
    if count <= 0 or offset < 0 or any(offset + count > size for size in sizes):
        raise ValueError(
            f"samples-per-domain={count}, offset={offset} is incompatible with {sizes}"
        )
    result = []
    start = 0
    for size in sizes:
        result.extend(
            suite["calibration"][start + offset : start + offset + count]
        )
        start += size
    return result


def target_modules(model, converted_layers):
    modules = {}
    for layer_index in range(len(model.model.layers)):
        if layer_index in converted_layers:
            continue
        for projection in PROJECTIONS:
            name = f"model.layers.{layer_index}.{projection}"
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"expected nn.Linear at {name}, found {type(module)}")
            modules[name] = module
    return modules


def summarize(scores: torch.Tensor, parameters: int):
    flat = scores.detach().float().flatten().cpu()
    return {
        "parameters": parameters,
        "groups": flat.numel(),
        "damage_mean": float(flat.mean()),
        "damage_sum": float(flat.sum()),
        "damage_p50": float(torch.quantile(flat, 0.50)),
        "damage_p90": float(torch.quantile(flat, 0.90)),
        "damage_p99": float(torch.quantile(flat, 0.99)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = (args.model_path or default_model_path()).resolve()
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    calibration = balanced_calibration(
        suite, args.samples_per_domain, args.domain_offset
    )
    seed_everything(args.seed)
    model = load_model(model_path, args.device)
    installed = install_checkpoint(model, source_payload, args.device)
    converted_layers = {
        int(name.split(".")[2])
        for name in installed
        if name.startswith("model.layers.")
    }
    modules = target_modules(model, converted_layers)
    collectors = {name: CausalMomentCollector(module) for name, module in modules.items()}

    # No model parameter needs a gradient. Making the embedding output a leaf
    # preserves a backward path through every decoder activation without
    # allocating multi-gigabyte weight-gradient buffers.
    anchor = model.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: output.detach().requires_grad_(True)
    )
    try:
        for index, chunk in enumerate(calibration):
            model.zero_grad(set_to_none=True)
            batch = chunk.unsqueeze(0).to(args.device)
            logits = model(input_ids=batch[:, :-1], use_cache=False).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                batch[:, 1:].reshape(-1),
            )
            loss.backward()
            log(f"sensitivity sample {index + 1}/{len(calibration)} loss={loss.item():.4f}")
    finally:
        anchor.remove()
        for collector in collectors.values():
            collector.close()

    module_results = {}
    with torch.no_grad():
        for index, (name, module) in enumerate(modules.items(), start=1):
            input_moment, output_moment = collectors[name].moments()
            scores = activation_fisher_group_damage(
                module.weight,
                input_moment,
                output_moment,
                group_size=args.group_size,
                relative=True,
            )
            module_results[name] = summarize(scores, module.weight.numel())
            if index % 14 == 0 or index == len(modules):
                log(f"scored {index}/{len(modules)} matrices")
            del scores, input_moment, output_moment

    block_results = {}
    for layer_index in range(len(model.model.layers)):
        names = [
            f"model.layers.{layer_index}.{projection}" for projection in PROJECTIONS
        ]
        if not all(name in module_results for name in names):
            continue
        total_groups = sum(module_results[name]["groups"] for name in names)
        total_damage = sum(module_results[name]["damage_sum"] for name in names)
        attention_names = [name for name in names if ".self_attn." in name]
        mlp_names = [name for name in names if ".mlp." in name]
        block_results[str(layer_index)] = {
            "parameters": sum(module_results[name]["parameters"] for name in names),
            "groups": total_groups,
            "damage_mean": total_damage / total_groups,
            "damage_sum": total_damage,
            "attention_damage_mean": sum(
                module_results[name]["damage_sum"] for name in attention_names
            )
            / sum(module_results[name]["groups"] for name in attention_names),
            "mlp_damage_mean": sum(
                module_results[name]["damage_sum"] for name in mlp_names
            )
            / sum(module_results[name]["groups"] for name in mlp_names),
        }

    ranking = sorted(
        (
            {
                "layer": int(layer),
                **values,
            }
            for layer, values in block_results.items()
        ),
        key=lambda item: item["damage_mean"],
    )
    result = {
        "schema": "wal-tat-decoder-sensitivity-v1",
        "model": str(model_path),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256_file(args.suite),
        "samples_per_domain": args.samples_per_domain,
        "domain_offset": args.domain_offset,
        "sample_count": len(calibration),
        "group_size": args.group_size,
        "converted_layers_excluded": sorted(converted_layers),
        "ranking": ranking,
        "blocks": block_results,
        "modules": module_results,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {output}; least-sensitive={[item['layer'] for item in ranking[:5]]}")
    del model, installed, collectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
