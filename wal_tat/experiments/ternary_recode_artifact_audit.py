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
from wal_tat.quantization import padded_grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, required=True)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.0001)
    parser.add_argument(
        "--cumulative-point-only",
        action="store_true",
        help=(
            "Require the cumulative observed ratio but use paired confidence "
            "only for the incremental candidate-vs-frontier comparison."
        ),
    )
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


def acceptance_decision(
    *,
    cumulative_point_passed: bool,
    cumulative_confidence_passed: bool,
    incremental_point_passed: bool,
    incremental_confidence_passed: bool,
    source_restoration_exact: bool,
    cumulative_point_only: bool,
) -> bool:
    return (
        cumulative_point_passed
        and (cumulative_point_only or cumulative_confidence_passed)
        and incremental_point_passed
        and incremental_confidence_passed
        and source_restoration_exact
    )


def artifact_matrix_entries(artifact: dict) -> dict[str, dict]:
    """Normalize legacy one-matrix and bundle recode artifacts."""
    artifact_format = artifact.get("format")
    if artifact_format == "wal-tat-ternary-recode-v1":
        return {
            artifact["target_name"]: {
                "committed_mask": artifact["committed_mask"],
                "ternary_codes_int8": artifact["ternary_codes_int8"],
                "scales_fp16": artifact["scales_fp16"],
            }
        }
    if artifact_format in {
        "wal-tat-ternary-recode-bundle-v1",
        "wal-tat-ternary-fallback-compensation-v1",
    }:
        entries = artifact.get("matrices")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("strict ternary recode bundle has no matrices")
        if set(entries) != set(artifact.get("target_names", ())):
            raise ValueError("bundle target_names do not match matrix entries")
        return entries
    raise ValueError("unsupported strict ternary recode artifact")


def validated_compensated_master(entry: dict, matrix, device: str):
    """Return a valid fallback-only BF16 master overlay, if present."""
    if "fp_master_bf16" not in entry:
        return None
    master = entry["fp_master_bf16"]
    if master.dtype != torch.bfloat16:
        raise ValueError("compensated master must be stored as BF16")
    if master.shape != matrix.master_weight.shape:
        raise ValueError("compensated master shape does not match frontier")
    if not torch.isfinite(master).all():
        raise ValueError("compensated master contains non-finite values")
    candidate = master.to(device).float()
    source_grouped, _, _ = padded_grouped(
        matrix.master_weight.detach(), matrix.group_size
    )
    candidate_grouped, _, _ = padded_grouped(candidate, matrix.group_size)
    if not torch.equal(
        candidate_grouped[matrix.committed_mask],
        source_grouped[matrix.committed_mask],
    ):
        raise ValueError("compensated master changes committed groups")
    return candidate


def artifact_code_churn(artifact: dict):
    """Read optional churn metadata without making it an audit prerequisite."""
    representation = artifact.get("representation", {})
    if "code_churn" in representation:
        return representation["code_churn"]
    total_change = artifact.get("total_change_from_checkpoint", {})
    return total_change.get("code_churn")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    artifact_entries = artifact_matrix_entries(artifact)
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

    target_names = tuple(artifact_entries)
    compensated_masters = {}
    for target_name, entry in artifact_entries.items():
        if target_name not in matrices:
            raise ValueError("artifact target is absent from source checkpoint")
        matrix = matrices[target_name]
        artifact_mask = entry["committed_mask"].to(args.device)
        codes = entry["ternary_codes_int8"].to(args.device)
        scales = entry["scales_fp16"].to(args.device).float()
        if not torch.equal(artifact_mask, matrix.committed_mask):
            raise ValueError("artifact committed mask does not match frontier")
        if codes.shape != matrix.committed_codes.shape:
            raise ValueError("artifact code shape does not match frontier")
        if scales.shape != matrix.group_scale.shape:
            raise ValueError("artifact scale shape does not match frontier")
        if not set(codes.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError("artifact contains non-ternary codes")
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("artifact contains invalid scales")
        if not torch.equal(
            codes[~artifact_mask], matrix.committed_codes[~artifact_mask]
        ):
            raise ValueError("artifact changes codes outside committed groups")
        if not torch.equal(scales[~artifact_mask], matrix.group_scale[~artifact_mask]):
            raise ValueError("artifact changes scales outside committed groups")
        compensated_master = validated_compensated_master(
            entry, matrix, args.device
        )
        if compensated_master is not None:
            compensated_masters[target_name] = compensated_master
        with torch.no_grad():
            matrix.committed_codes.copy_(codes)
            matrix.group_scale.copy_(scales)
            if compensated_master is not None:
                matrix.master_weight.copy_(compensated_master)
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
        "artifact_format": artifact["format"],
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "target_names": list(target_names),
        "recipe": artifact["recipe"],
        "residual_gain": artifact.get("residual_gain", 1.0),
        "group_sizes": {
            name: matrices[name].group_size for name in target_names
        },
        "code_churn": artifact_code_churn(artifact),
        "fallback_relative_delta": artifact.get("aggregate_relative_delta"),
        "compensated_master_names": sorted(compensated_masters),
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
        "cumulative_acceptance_mode": (
            "point_only" if args.cumulative_point_only else "point_and_confidence"
        ),
        "accepted": acceptance_decision(
            cumulative_point_passed=cumulative_point_passed,
            cumulative_confidence_passed=cumulative_confidence_passed,
            incremental_point_passed=incremental_point_passed,
            incremental_confidence_passed=incremental_confidence_passed,
            source_restoration_exact=source_restoration_exact,
            cumulative_point_only=args.cumulative_point_only,
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
