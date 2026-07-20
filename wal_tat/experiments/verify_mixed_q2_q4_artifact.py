"""Fresh-process verification for a strict mixed Q2-g128/Q4-g128 artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    set_submodule,
    sha256_file,
)
from wal_tat import accepted_weight_counts, q2_g128_physical_bpw, q4_g128_physical_bpw


class FixedMixedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.bias = None if bias is None else nn.Parameter(
            bias.detach().clone(), requires_grad=False
        )
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight.to(value.dtype), self.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument("--total-major-weights", type=int, default=1_720_451_072)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def valid_weight_count(mask: torch.Tensor, columns: int, group_size: int) -> int:
    full_groups, remainder = divmod(columns, group_size)
    count = int(mask[:, :full_groups].sum().item()) * group_size
    if remainder:
        count += int(mask[:, full_groups].sum().item()) * remainder
    return count


def main() -> None:
    args = parse_args()
    source_path = args.source_checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    if artifact.get("format") != "wal-tat-mixed-q2-q4-v1":
        raise ValueError("unsupported mixed artifact format")
    if artifact.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("mixed artifact source checkpoint mismatch")
    if artifact.get("group_size") != 128:
        raise ValueError("mixed artifact group size must be 128")
    if artifact.get("q2_physical_bpw") != q2_g128_physical_bpw():
        raise ValueError("mixed artifact Q2 format mismatch")
    if artifact.get("q4_physical_bpw") != q4_g128_physical_bpw():
        raise ValueError("mixed artifact Q4 format mismatch")

    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, suite["gates"], args.device)
    matrices = install_checkpoint(model, source, args.device)
    source_metrics = evaluate_domains(model, suite["gates"], args.device)
    source_ratios = ratios(source_metrics, baseline)
    source_ternary_counts = accepted_weight_counts(source)
    source_ternary_weights = sum(source_ternary_counts.values())
    new_q2_weights = 0
    q4_weights = 0
    matrix_statistics = {}
    for name, entry in artifact["matrices"].items():
        if name not in matrices or name not in source["matrices"]:
            raise ValueError(f"artifact matrix is absent from source: {name}")
        matrix = matrices[name]
        shape = tuple(entry["shape"])
        if shape != (matrix.out_features, matrix.in_features):
            raise ValueError(f"artifact matrix shape mismatch for {name}")
        source_mask = entry["source_committed_mask"].bool()
        checkpoint_mask = source["matrices"][name]["committed_mask"].bool()
        if not torch.equal(source_mask, checkpoint_mask):
            raise ValueError(f"source committed mask mismatch for {name}")
        q4_mask = entry["q4_mask"].bool()
        q2_codes = entry["q2_codes_int8"].to(torch.int8)
        q2_scales = entry["q2_scales_fp16"].half()
        q4_codes = entry["q4_codes_int8"].to(torch.int8)
        q4_scales = entry["q4_scales_fp16"].half()
        expected_code_shape = tuple(matrix.committed_codes.shape)
        expected_scale_shape = tuple(matrix.group_scale.shape)
        if tuple(q2_codes.shape) != expected_code_shape or tuple(q4_codes.shape) != expected_code_shape:
            raise ValueError(f"artifact code shape mismatch for {name}")
        if tuple(q2_scales.shape) != expected_scale_shape or tuple(q4_scales.shape) != expected_scale_shape:
            raise ValueError(f"artifact scale shape mismatch for {name}")
        if tuple(q4_mask.shape) != expected_scale_shape:
            raise ValueError(f"artifact mask shape mismatch for {name}")
        if not set(q2_codes.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError(f"non-ternary Q2 code in {name}")
        if not all(-8 <= value <= 7 for value in q4_codes.unique().tolist()):
            raise ValueError(f"out-of-range signed Q4 code in {name}")
        if torch.any(q4_mask & source_mask):
            raise ValueError(f"Q4 rescue overwrites an accepted ternary group in {name}")
        source_codes = source["matrices"][name]["ternary_codes_int8"].to(
            torch.int8
        )
        source_codes = F.pad(
            source_codes, (0, matrix.padding)
        ).view_as(matrix.committed_codes)
        source_scales = source["matrices"][name]["scales_fp16"].half()
        if not torch.equal(q2_codes[source_mask], source_codes[source_mask]):
            raise ValueError(f"accepted source codes changed in {name}")
        if not torch.equal(q2_scales[source_mask], source_scales[source_mask]):
            raise ValueError(f"accepted source scales changed in {name}")
        q2_value = q2_codes.float() * q2_scales.float().unsqueeze(-1)
        q4_value = q4_codes.float() * q4_scales.float().unsqueeze(-1)
        grouped = torch.where(q4_mask.unsqueeze(-1), q4_value, q2_value)
        weight = grouped.reshape(matrix.out_features, -1)[:, : matrix.in_features]
        target = model.get_submodule(name)
        set_submodule(
            model,
            name,
            FixedMixedLinear(weight.to(matrix.compute_dtype), target.bias).to(args.device),
        )
        new_q2_mask = (~source_mask) & (~q4_mask)
        matrix_new_q2 = valid_weight_count(new_q2_mask, matrix.in_features, 128)
        matrix_q4 = valid_weight_count(q4_mask, matrix.in_features, 128)
        new_q2_weights += matrix_new_q2
        q4_weights += matrix_q4
        matrix_statistics[name] = {
            "source_ternary_weights": source_ternary_counts[name],
            "new_ternary_weights": matrix_new_q2,
            "q4_weights": matrix_q4,
        }

    metrics = evaluate_domains(model, suite["gates"], args.device)
    metric_ratios = ratios(metrics, baseline)
    incremental_ratios = ratios(metrics, source_metrics)
    cumulative_passed = all(
        value <= args.gate_ratio for value in metric_ratios.values()
    )
    incremental_passed = all(
        value <= args.incremental_gate_ratio
        for value in incremental_ratios.values()
    )
    passed = cumulative_passed and incremental_passed
    strict_ternary_weights = source_ternary_weights + new_q2_weights
    low_bit_weights = strict_ternary_weights + q4_weights
    result = {
        "schema": "wal-tat-verify-mixed-q2-q4-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "baseline": baseline,
        "source_metrics": source_metrics,
        "source_ratios": source_ratios,
        "metrics": metrics,
        "ratios": metric_ratios,
        "incremental_ratios_vs_source": incremental_ratios,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "cumulative_passed": cumulative_passed,
        "incremental_passed": incremental_passed,
        "passed": passed,
        "strict_q2_codes_only": True,
        "signed_q4_codes_only": True,
        "source_ternary_weights": source_ternary_weights,
        "new_ternary_weights": new_q2_weights,
        "strict_ternary_weights": strict_ternary_weights,
        "q4_weights": q4_weights,
        "low_bit_weights": low_bit_weights,
        "strict_ternary_coverage": strict_ternary_weights / args.total_major_weights,
        "q4_coverage": q4_weights / args.total_major_weights,
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
