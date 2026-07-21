"""Fresh-process verification for a strict mixed Q2-g128/Q4-g128 artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    sha256_file,
)
from wal_tat import accepted_weight_counts, install_mixed_q2_q4_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--parent-artifact",
        type=Path,
        help="Optional accepted mixed parent for a direct incremental gate.",
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument(
        "--source-ppl-ratio",
        type=float,
        help=(
            "Optional cumulative perplexity-ratio limit versus the immutable "
            "strict source. When supplied, this replaces the source-relative "
            "NLL-ratio acceptance gate."
        ),
    )
    parser.add_argument(
        "--parent-ppl-ratio",
        type=float,
        help=(
            "Optional perplexity-ratio limit versus the accepted parent. "
            "When supplied, this replaces the parent-relative NLL-ratio gate."
        ),
    )
    parser.add_argument("--total-major-weights", type=int, default=1_720_451_072)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def perplexity_ratios(candidate: dict, reference: dict) -> dict[str, float]:
    """Return exact per-domain PPL ratios from additive mean-NLL changes."""
    if candidate.keys() != reference.keys():
        raise ValueError("candidate and reference metric domains must match")
    return {
        domain: math.exp(candidate[domain]["nll"] - reference[domain]["nll"])
        for domain in candidate
    }


def ratio_gate_passed(values: dict[str, float], limit: float) -> bool:
    if limit < 1.0:
        raise ValueError("quality-gate ratio must be at least 1.0")
    return all(value <= limit for value in values.values())


def main() -> None:
    args = parse_args()
    source_path = args.source_checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    parent_artifact_path = (
        args.parent_artifact.resolve() if args.parent_artifact is not None else None
    )
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    parent_artifact = (
        torch.load(parent_artifact_path, map_location="cpu", weights_only=False)
        if parent_artifact_path is not None
        else None
    )
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, suite["gates"], args.device)
    matrices = install_checkpoint(model, source, args.device)
    source_metrics = evaluate_domains(model, suite["gates"], args.device)
    source_ratios = ratios(source_metrics, baseline)
    source_ternary_counts = accepted_weight_counts(source)
    source_ternary_weights = sum(source_ternary_counts.values())
    parent_metrics = None
    parent_incremental_ratios = None
    if parent_artifact is not None:
        install_mixed_q2_q4_artifact(
            model,
            parent_artifact,
            source,
            device=args.device,
            expected_source_sha256=sha256_file(source_path),
        )
        parent_metrics = evaluate_domains(model, suite["gates"], args.device)
    install_result = install_mixed_q2_q4_artifact(
        model,
        artifact,
        source,
        device=args.device,
        expected_source_sha256=sha256_file(source_path),
    )
    new_q2_weights = install_result.new_q2_weights
    q4_weights = install_result.q4_weights
    q8_weights = install_result.q8_weights
    matrix_statistics = install_result.matrix_statistics

    metrics = evaluate_domains(model, suite["gates"], args.device)
    metric_ratios = ratios(metrics, baseline)
    incremental_ratios = ratios(metrics, source_metrics)
    source_ppl_ratios = perplexity_ratios(metrics, source_metrics)
    parent_ppl_ratios = None
    if parent_metrics is not None:
        parent_incremental_ratios = ratios(metrics, parent_metrics)
        parent_ppl_ratios = perplexity_ratios(metrics, parent_metrics)
    cumulative_passed = ratio_gate_passed(metric_ratios, args.gate_ratio)
    source_gate_metric = (
        source_ppl_ratios
        if args.source_ppl_ratio is not None
        else incremental_ratios
    )
    source_gate_limit = (
        args.source_ppl_ratio
        if args.source_ppl_ratio is not None
        else args.incremental_gate_ratio
    )
    incremental_passed = ratio_gate_passed(source_gate_metric, source_gate_limit)
    parent_gate_metric = (
        parent_ppl_ratios
        if args.parent_ppl_ratio is not None
        else parent_incremental_ratios
    )
    parent_gate_limit = (
        args.parent_ppl_ratio
        if args.parent_ppl_ratio is not None
        else args.incremental_gate_ratio
    )
    parent_incremental_passed = parent_gate_metric is None or ratio_gate_passed(
        parent_gate_metric, parent_gate_limit
    )
    passed = cumulative_passed and incremental_passed and parent_incremental_passed
    strict_ternary_weights = source_ternary_weights + new_q2_weights
    low_bit_weights = strict_ternary_weights + q4_weights + q8_weights
    result = {
        "schema": "wal-tat-verify-mixed-q2-q4-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "parent_artifact": (
            str(parent_artifact_path) if parent_artifact_path is not None else None
        ),
        "parent_artifact_sha256": (
            sha256_file(parent_artifact_path)
            if parent_artifact_path is not None
            else None
        ),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "baseline": baseline,
        "source_metrics": source_metrics,
        "source_ratios": source_ratios,
        "metrics": metrics,
        "ratios": metric_ratios,
        "incremental_ratios_vs_source": incremental_ratios,
        "ppl_ratios_vs_source": source_ppl_ratios,
        "parent_metrics": parent_metrics,
        "incremental_ratios_vs_parent": parent_incremental_ratios,
        "ppl_ratios_vs_parent": parent_ppl_ratios,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "source_gate_metric": (
            "ppl_ratio" if args.source_ppl_ratio is not None else "nll_ratio"
        ),
        "source_gate_limit": source_gate_limit,
        "parent_gate_metric": (
            "ppl_ratio" if args.parent_ppl_ratio is not None else "nll_ratio"
        ),
        "parent_gate_limit": parent_gate_limit,
        "cumulative_passed": cumulative_passed,
        "incremental_passed": incremental_passed,
        "parent_incremental_passed": parent_incremental_passed,
        "passed": passed,
        "strict_q2_codes_only": True,
        "signed_q4_codes_only": True,
        "signed_q8_codes_only": True,
        "source_ternary_weights": source_ternary_weights,
        "new_ternary_weights": new_q2_weights,
        "strict_ternary_weights": strict_ternary_weights,
        "q4_weights": q4_weights,
        "q8_weights": q8_weights,
        "low_bit_weights": low_bit_weights,
        "strict_ternary_coverage": strict_ternary_weights / args.total_major_weights,
        "q4_coverage": q4_weights / args.total_major_weights,
        "q8_coverage": q8_weights / args.total_major_weights,
        "low_bit_coverage": low_bit_weights / args.total_major_weights,
        "remaining_non_low_bit_weights": args.total_major_weights - low_bit_weights,
        "matrix_statistics": matrix_statistics,
        "checkpoint_written": False,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; passed={passed} ratios={metric_ratios}", flush=True)


if __name__ == "__main__":
    main()
