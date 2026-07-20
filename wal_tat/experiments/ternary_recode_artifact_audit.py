"""Paired fresh-process audit for a frozen strict ternary recode artifact."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from compensation_window import (
    PROJECT,
    default_model_path,
    install_checkpoint,
    load_model,
    seed_everything,
    sha256_file,
)
from counterfactual_bf16_restore_audit import restore_source_state
from rht_artifact_audit import aggregate_ratio, evaluate_window_sums, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, required=True)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.0001)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--bootstrap-block-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument(
        "--suite-role",
        default="recurring validation; frozen artifact; no gradient updates",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("format") != "wal-tat-ternary-recode-v1":
        raise ValueError("unsupported strict ternary recode artifact")
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("artifact source checkpoint hash does not match")
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
    if target_name not in matrices:
        raise ValueError("artifact target is absent from source checkpoint")
    matrix = matrices[target_name]
    artifact_mask = artifact["committed_mask"].to(args.device)
    codes = artifact["ternary_codes_int8"].to(args.device)
    scales = artifact["scales_fp16"].to(args.device).float()
    if not torch.equal(artifact_mask, matrix.committed_mask):
        raise ValueError("artifact committed mask does not match frontier")
    if codes.shape != matrix.committed_codes.shape:
        raise ValueError("artifact code shape does not match frontier")
    if scales.shape != matrix.group_scale.shape:
        raise ValueError("artifact scale shape does not match frontier")
    if not set(codes.unique().tolist()) <= {-1, 0, 1}:
        raise ValueError("artifact contains non-ternary codes")
    if not torch.equal(
        codes[~artifact_mask], matrix.committed_codes[~artifact_mask]
    ):
        raise ValueError("artifact changes codes outside committed groups")
    if not torch.equal(scales[~artifact_mask], matrix.group_scale[~artifact_mask]):
        raise ValueError("artifact changes scales outside committed groups")

    with torch.no_grad():
        matrix.committed_codes.copy_(codes)
        matrix.group_scale.copy_(scales)
    candidate = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }

    cumulative_domains = {
        domain: paired(candidate[domain], bf16[domain], args, args.gate_ratio)
        for domain in gates
    }
    incremental_domains = {
        domain: paired(
            candidate[domain],
            frontier[domain],
            args,
            args.incremental_gate_ratio,
        )
        for domain in gates
    }
    restore_source_state(model, matrices, source_payload, args.device)
    frontier_recheck = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }
    source_restoration_exact = all(
        frontier_recheck[domain]["nll_sums"] == frontier[domain]["nll_sums"]
        for domain in gates
    )
    cumulative_point_passed = all(
        value["point_passed"] for value in cumulative_domains.values()
    )
    cumulative_confidence_passed = all(
        value["confidence_passed"] for value in cumulative_domains.values()
    )
    incremental_point_passed = all(
        value["point_passed"] for value in incremental_domains.values()
    )
    incremental_confidence_passed = all(
        value["confidence_passed"] for value in incremental_domains.values()
    )
    result = {
        "schema": "wal-tat-ternary-recode-artifact-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "target_name": target_name,
        "recipe": artifact["recipe"],
        "residual_gain": artifact["residual_gain"],
        "group_size": matrix.group_size,
        "code_churn": artifact["representation"]["code_churn"],
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "frontier_ratios_to_bf16": {
            domain: aggregate_ratio(frontier[domain], bf16[domain])
            for domain in gates
        },
        "candidate_ratios_to_bf16": {
            domain: aggregate_ratio(candidate[domain], bf16[domain])
            for domain in gates
        },
        "cumulative_candidate_vs_bf16": cumulative_domains,
        "incremental_candidate_vs_frontier": incremental_domains,
        "cumulative_point_passed": cumulative_point_passed,
        "cumulative_confidence_passed": cumulative_confidence_passed,
        "incremental_point_passed": incremental_point_passed,
        "incremental_confidence_passed": incremental_confidence_passed,
        "accepted": (
            cumulative_point_passed
            and cumulative_confidence_passed
            and incremental_point_passed
            and incremental_confidence_passed
            and source_restoration_exact
        ),
        "source_restoration_exact": source_restoration_exact,
        "checkpoint_mutated": sha256_file(checkpoint) != checkpoint_hash,
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
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "raw_window_losses"},
            indent=2,
        )
    )
    if not source_restoration_exact:
        raise SystemExit(2)
    del model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
