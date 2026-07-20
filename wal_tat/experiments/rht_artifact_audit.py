"""Fresh cumulative BF16 audit for a fixed RHT ternary artifact."""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from wal_tat import FixedTernaryLinear, paired_block_bootstrap_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-block-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument(
        "--suite-role",
        default="recurring validation; no gradient updates",
        help="Declared methodological role of this suite in the current study.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_window_sums(model, chunks, device: str, batch_size: int) -> dict:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    model.eval()
    nll_sums: list[float] = []
    token_counts: list[int] = []
    for start in range(0, len(chunks), batch_size):
        selected = chunks[start : start + batch_size]
        if len({int(chunk.numel()) for chunk in selected}) != 1:
            raise ValueError("batched audit windows must have equal lengths")
        batch = torch.stack(selected).to(device)
        logits = model(input_ids=batch[:, :-1], use_cache=False).logits
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], -1)
        nll_sums.extend(float(value) for value in losses.sum(dim=1).cpu())
        token_counts.extend([batch.shape[1] - 1] * batch.shape[0])
    total_nll = sum(nll_sums)
    total_tokens = sum(token_counts)
    mean_nll = total_nll / total_tokens
    return {
        "nll_sums": nll_sums,
        "token_counts": token_counts,
        "aggregate": {
            "nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 50.0)),
            "tokens": total_tokens,
        },
    }


def aggregate_ratio(candidate: dict, baseline: dict) -> float:
    return candidate["aggregate"]["nll"] / baseline["aggregate"]["nll"]


def paired(candidate: dict, baseline: dict, args, gate_ratio: float) -> dict:
    return paired_block_bootstrap_nll(
        baseline["nll_sums"],
        candidate["nll_sums"],
        baseline["token_counts"],
        gate_ratio=gate_ratio,
        samples=args.bootstrap_samples,
        block_size=args.bootstrap_block_size,
        confidence=args.confidence,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "wal-tat-rht-ternary-artifact-v1":
        raise ValueError("unsupported transform artifact schema")
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("artifact source checkpoint hash does not match")
    if artifact.get("development_only") is not True:
        raise ValueError("artifact must be marked development_only before audit")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    gates = suite["gates"]
    seed_everything(args.seed)

    model = load_model(model_path, args.device)
    bf16 = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, source_payload, args.device)
    frontier = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }

    target_name = artifact["target_name"]
    target = model.get_submodule(target_name)
    if not isinstance(target, nn.Linear):
        raise TypeError("artifact target is not an untouched BF16 nn.Linear")
    codes = artifact["codes"]
    scales = artifact["scales"]
    if list(target.weight.shape) != artifact["target_shape"]:
        raise ValueError("artifact target shape does not match model")
    if codes.ndim != 3 or scales.shape != codes.shape[:2]:
        raise ValueError("artifact codes/scales have invalid shapes")
    bias = artifact.get("bias")
    layer = FixedTernaryLinear(
        codes,
        scales,
        compute_dtype=target.weight.dtype,
        transform=artifact["transform"],
        transform_seed=int(artifact["transform_seed"]),
        bias=bias,
    ).to(args.device)
    set_submodule(model, target_name, layer)
    candidate = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }

    cumulative_domains = {
        domain: paired(candidate[domain], bf16[domain], args, args.gate_ratio)
        for domain in gates
    }
    incremental_domains = {
        domain: paired(candidate[domain], frontier[domain], args, 1.0)
        for domain in gates
    }
    frontier_ratios_to_bf16 = {
        domain: aggregate_ratio(frontier[domain], bf16[domain])
        for domain in gates
    }
    result = {
        "schema": "wal-tat-rht-artifact-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "target_name": target_name,
        "transform": artifact["transform"],
        "transform_seed": artifact["transform_seed"],
        "group_size": artifact["group_size"],
        "gate_ratio": args.gate_ratio,
        "frontier_ratios_to_bf16": frontier_ratios_to_bf16,
        "cumulative_candidate_vs_bf16": cumulative_domains,
        "incremental_candidate_vs_frontier": incremental_domains,
        "point_passed": all(
            value["point_passed"] for value in cumulative_domains.values()
        ),
        "confidence_passed": all(
            value["confidence_passed"] for value in cumulative_domains.values()
        ),
        "worst_cumulative_observed_ratio": max(
            value["observed_ratio"] for value in cumulative_domains.values()
        ),
        "worst_cumulative_upper_ratio": max(
            value["ratio_ci"]["upper"] for value in cumulative_domains.values()
        ),
        "checkpoint_mutated": False,
        "accepted_coverage_changed": False,
        "raw_window_losses": {
            domain: {
                "bf16_nll_sums": bf16[domain]["nll_sums"],
                "frontier_nll_sums": frontier[domain]["nll_sums"],
                "candidate_nll_sums": candidate[domain]["nll_sums"],
                "token_counts": bf16[domain]["token_counts"],
            }
            for domain in gates
        },
        "existing_frontier_coverage": {
            name: float(matrix.committed_mask.float().mean().item())
            for name, matrix in matrices.items()
        },
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "raw_window_losses"},
            indent=2,
        )
    )

    del model, matrices, layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
