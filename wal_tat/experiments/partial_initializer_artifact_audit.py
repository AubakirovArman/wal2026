"""Paired cumulative audit for a frozen partial ternary initializer artifact."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    default_model_path,
    install_checkpoint,
    load_model,
    seed_everything,
    sha256_file,
)
from rht_artifact_audit import aggregate_ratio, evaluate_window_sums, paired
from wal_tat import hard_codes_scales


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
        "--entry-decile",
        type=int,
        action="append",
        default=[],
        help="Audit only the requested artifact decile(s); repeat as needed.",
    )
    parser.add_argument(
        "--suite-role",
        default="recurring validation; artifact frozen on development",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def grouped_codes(codes: torch.Tensor, matrix) -> torch.Tensor:
    value = codes
    if matrix.padding:
        value = F.pad(value, (0, matrix.padding))
    return value.view_as(matrix.committed_codes)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "wal-tat-partial-initializer-artifact-v1":
        raise ValueError("unsupported partial initializer artifact schema")
    if artifact.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("artifact source checkpoint hash does not match")
    if artifact.get("development_only") is not True:
        raise ValueError("artifact must have been frozen on development only")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    gates = suite["gates"]
    seed_everything(args.seed)

    model = load_model(model_path, args.device)
    bf16 = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    matrices = install_checkpoint(model, payload, args.device)
    frontier = {
        domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
        for domain, chunks in gates.items()
    }
    target_name = artifact["target_name"]
    if target_name not in matrices:
        raise ValueError("artifact target is absent from the partial frontier")
    matrix = matrices[target_name]
    if list(matrix.master_weight.shape) != artifact["target_shape"]:
        raise ValueError("artifact target shape does not match")
    if int(artifact["group_size"]) != matrix.group_size:
        raise ValueError("artifact group size does not match")

    base_codes, base_scales = hard_codes_scales(
        matrix.master_weight,
        matrix.group_size,
        matrix.group_scale,
    )
    codes = grouped_codes(base_codes, matrix)
    scales = base_scales.clone()
    mask = torch.zeros_like(matrix.committed_mask)
    flat_mask = mask.reshape(-1)
    flat_codes = codes.reshape(-1, matrix.group_size)
    flat_scales = scales.reshape(-1)
    requested_deciles = tuple(args.entry_decile)
    if len(set(requested_deciles)) != len(requested_deciles):
        raise ValueError("entry-decile values must be unique")
    if any(not 1 <= value <= 10 for value in requested_deciles):
        raise ValueError("entry-decile must be in [1, 10]")
    selected_entries = [
        entry
        for entry in artifact["entries"]
        if not requested_deciles or int(entry["decile"]) in requested_deciles
    ]
    if not selected_entries:
        raise ValueError("no artifact entries matched entry-decile")
    entry_summary = []
    for entry in selected_entries:
        indices = entry["flat_group_indices"].to(args.device, dtype=torch.long)
        if torch.any(indices < 0) or torch.any(indices >= flat_mask.numel()):
            raise ValueError("artifact group index is out of bounds")
        if torch.any(matrix.committed_mask.reshape(-1).index_select(0, indices)):
            raise ValueError("artifact overlaps already committed groups")
        if torch.any(flat_mask.index_select(0, indices)):
            raise ValueError("artifact entries overlap each other")
        entry_codes = entry["codes"].to(args.device, dtype=torch.int8)
        entry_scales = entry["scales"].to(args.device, dtype=flat_scales.dtype)
        if entry_codes.shape != (indices.numel(), matrix.group_size):
            raise ValueError("artifact code shape does not match indices")
        if entry_scales.shape != (indices.numel(),):
            raise ValueError("artifact scale shape does not match indices")
        flat_mask[indices] = True
        flat_codes[indices] = entry_codes
        flat_scales[indices] = entry_scales
        entry_summary.append(
            {
                "decile": int(entry["decile"]),
                "initializer": entry["initializer"],
                "groups": int(indices.numel()),
                "weights": int(indices.numel() * matrix.group_size),
                "development_worst_incremental_ratio": float(
                    entry["development_worst_incremental_ratio"]
                ),
            }
        )

    matrix.begin(mask, transaction_id="partial-initializer-artifact-audit")
    try:
        matrix.set_candidate_codes(codes, scales)
        matrix.set_candidate_state(1.0, 0.0)
        candidate = {
            domain: evaluate_window_sums(model, chunks, args.device, args.batch_size)
            for domain, chunks in gates.items()
        }
    finally:
        matrix.rollback()

    cumulative_domains = {
        domain: paired(candidate[domain], bf16[domain], args, args.gate_ratio)
        for domain in gates
    }
    incremental_domains = {
        domain: paired(candidate[domain], frontier[domain], args, 1.0)
        for domain in gates
    }
    result = {
        "schema": "wal-tat-partial-initializer-artifact-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "suite_role": args.suite_role,
        "model": str(model_path),
        "target_name": target_name,
        "group_size": matrix.group_size,
        "entries": entry_summary,
        "requested_deciles": list(requested_deciles),
        "candidate_groups": int(mask.sum().item()),
        "candidate_weights": int(mask.sum().item() * matrix.group_size),
        "gate_ratio": args.gate_ratio,
        "frontier_ratios_to_bf16": {
            domain: aggregate_ratio(frontier[domain], bf16[domain]) for domain in gates
        },
        "cumulative_candidate_vs_bf16": cumulative_domains,
        "incremental_candidate_vs_frontier": incremental_domains,
        "point_passed": all(value["point_passed"] for value in cumulative_domains.values()),
        "confidence_passed": all(
            value["confidence_passed"] for value in cumulative_domains.values()
        ),
        "worst_cumulative_observed_ratio": max(
            value["observed_ratio"] for value in cumulative_domains.values()
        ),
        "worst_cumulative_upper_ratio": max(
            value["ratio_ci"]["upper"] for value in cumulative_domains.values()
        ),
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
    print(json.dumps({key: value for key, value in result.items() if key != "raw_window_losses"}, indent=2))
    del model, matrices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
