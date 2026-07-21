"""Project the tied Qwen embedding/output head to one deployable low-bit format.

The decoder parent is immutable.  The tied matrix is represented once in the
artifact and reinstalled as one shared tensor in both ``embed_tokens`` and
``lm_head``.  This script deliberately evaluates one requested precision at a
time so a failed Q2 or Q4 candidate cannot be mistaken for an accepted stage.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    sha256_file,
)
from wal_tat import (
    accepted_weight_counts,
    install_mixed_q2_q4_artifact,
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
    weighted_symmetric_q4_project,
    weighted_symmetric_q8_project,
)
from wal_tat.scoring import exact_diagonal_ternary_project


EMBEDDING_NAME = "model.embed_tokens"
HEAD_NAME = "lm_head"
GROUP_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--precision", choices=("q2", "q4", "q8"), required=True)
    parser.add_argument("--moment-sequences", type=int, default=64)
    parser.add_argument("--projection-row-chunk", type=int, default=1024)
    parser.add_argument(
        "--head-moment-weight",
        type=float,
        default=0.8,
        help="blend normalized output-head moments with uniform embedding moments",
    )
    parser.add_argument("--gate-ratio", type=float, default=1.1)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.1)
    parser.add_argument("--total-major-weights", type=int, default=1_720_451_072)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=647)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def collect_head_input_moment(
    model: nn.Module,
    chunks: list[torch.Tensor],
    *,
    device: str,
    count: int,
) -> torch.Tensor:
    if not 1 <= count <= len(chunks):
        raise ValueError("moment sequence count is outside calibration suite")
    total = None
    tokens = 0
    for chunk in chunks[:count]:
        batch = chunk.unsqueeze(0).to(device)
        hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state.float()
        squared = hidden.square().sum((0, 1))
        total = squared if total is None else total + squared
        tokens += hidden.shape[0] * hidden.shape[1]
    if total is None or tokens == 0:
        raise RuntimeError("no hidden states collected")
    return (total / tokens).clamp_min(1e-12)


@torch.no_grad()
def project_rows(
    weight: torch.Tensor,
    moment: torch.Tensor,
    *,
    precision: str,
    row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if row_chunk < 1:
        raise ValueError("projection row chunk must be positive")
    if precision not in {"q2", "q4", "q8"}:
        raise ValueError(f"unsupported precision: {precision}")
    projector = {
        "q2": exact_diagonal_ternary_project,
        "q4": weighted_symmetric_q4_project,
        "q8": weighted_symmetric_q8_project,
    }[precision]
    code_chunks = []
    scale_chunks = []
    error_sum = 0.0
    error_max = 0.0
    group_count = 0
    for start in range(0, weight.shape[0], row_chunk):
        stop = min(start + row_chunk, weight.shape[0])
        codes, scales, error = projector(
            weight[start:stop].float(), moment, group_size=GROUP_SIZE
        )
        code_chunks.append(codes.to("cpu", dtype=torch.int8))
        scale_chunks.append(scales.to("cpu", dtype=torch.float16))
        error_sum += float(error.double().sum().item())
        error_max = max(error_max, float(error.max().item()))
        group_count += error.numel()
        if stop == weight.shape[0] or stop % (row_chunk * 32) == 0:
            print(f"projected tied rows {stop}/{weight.shape[0]}", flush=True)
    return (
        torch.cat(code_chunks),
        torch.cat(scale_chunks),
        {
            "weighted_error_sum": error_sum,
            "weighted_error_mean_per_group": error_sum / max(group_count, 1),
            "weighted_error_max_group": error_max,
        },
    )


def grouped_codes(codes: torch.Tensor, *, rows: int, columns: int) -> torch.Tensor:
    if tuple(codes.shape) == (rows, columns):
        return codes.view(rows, columns // GROUP_SIZE, GROUP_SIZE)
    expected = (rows, columns // GROUP_SIZE, GROUP_SIZE)
    if tuple(codes.shape) != expected:
        raise ValueError(f"projected code shape {tuple(codes.shape)} != {expected}")
    return codes


def main() -> None:
    args = parse_args()
    if not 0 <= args.head_moment_weight <= 1:
        raise ValueError("head moment weight must be in [0, 1]")
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    seed_everything(args.seed)
    started = time.time()

    model = load_model(model_path, args.device)
    embedding = model.get_submodule(EMBEDDING_NAME)
    head = model.get_submodule(HEAD_NAME)
    if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
        raise TypeError("expected a tied embedding and linear output head")
    if embedding.weight is not head.weight:
        raise ValueError("model embedding and output head are not tied")
    rows, columns = map(int, embedding.weight.shape)
    if columns % GROUP_SIZE:
        raise ValueError("tied matrix columns are not divisible by g128")

    baseline = evaluate_domains(model, gates, args.device)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    install_checkpoint(model, source, args.device)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    install_mixed_q2_q4_artifact(
        model,
        parent,
        source,
        device=args.device,
        expected_source_sha256=sha256_file(source_path),
    )
    parent_metrics = evaluate_domains(model, gates, args.device)

    embedding = model.get_submodule(EMBEDDING_NAME)
    head = model.get_submodule(HEAD_NAME)
    if embedding.weight is not head.weight:
        raise ValueError("parent installation broke the raw embedding/head tie")
    weight = embedding.weight.detach()
    head_moment = collect_head_input_moment(
        model,
        calibration,
        device=args.device,
        count=args.moment_sequences,
    )
    normalized_head_moment = head_moment / head_moment.mean().clamp_min(1e-12)
    projection_moment = torch.lerp(
        torch.ones_like(normalized_head_moment),
        normalized_head_moment,
        args.head_moment_weight,
    )
    codes, scales, projection_statistics = project_rows(
        weight,
        projection_moment,
        precision=args.precision,
        row_chunk=args.projection_row_chunk,
    )
    codes = grouped_codes(codes, rows=rows, columns=columns)
    groups = columns // GROUP_SIZE
    mask_shape = (rows, groups)
    code_shape = (rows, groups, GROUP_SIZE)
    all_groups = torch.ones(mask_shape, dtype=torch.bool)
    no_groups = torch.zeros(mask_shape, dtype=torch.bool)
    zero_codes = torch.zeros(code_shape, dtype=torch.int8)
    unit_scales = torch.ones(mask_shape, dtype=torch.float16)
    entry = {
        "kind": "tied_embedding_head",
        "tied_linear_name": HEAD_NAME,
        "shape": (rows, columns),
        "source_committed_mask": no_groups,
        "q4_mask": all_groups if args.precision == "q4" else no_groups,
        "q8_mask": all_groups if args.precision == "q8" else no_groups,
        "q2_codes_int8": codes if args.precision == "q2" else zero_codes,
        "q2_scales_fp16": scales if args.precision == "q2" else unit_scales,
        "q4_codes_int8": codes if args.precision == "q4" else zero_codes,
        "q4_scales_fp16": scales if args.precision == "q4" else unit_scales,
        "q8_codes_int8": codes if args.precision == "q8" else zero_codes,
        "q8_scales_fp16": scales if args.precision == "q8" else unit_scales,
    }
    matrices = dict(parent["matrices"])
    if EMBEDDING_NAME in matrices or HEAD_NAME in matrices:
        raise ValueError("parent already contains the tied matrix")
    matrices[EMBEDDING_NAME] = entry
    artifact = dict(parent)
    artifact.update(
        {
            "format": "wal-tat-mixed-q2-q4-q8-v1",
            "source_checkpoint_sha256": sha256_file(source_path),
            "suite_sha256": sha256_file(suite_path),
            "parent_artifact_sha256": sha256_file(parent_path),
            "target_kind": "tied_embedding_head",
            "target_precision": args.precision,
            "group_size": GROUP_SIZE,
            "q2_physical_bpw": q2_g128_physical_bpw(),
            "q4_physical_bpw": q4_g128_physical_bpw(),
            "q8_physical_bpw": q8_g128_physical_bpw(),
            "matrices": matrices,
        }
    )
    install_result = install_mixed_q2_q4_artifact(
        model,
        artifact,
        source,
        device=args.device,
        expected_source_sha256=sha256_file(source_path),
    )
    installed_embedding = model.get_submodule(EMBEDDING_NAME)
    installed_head = model.get_submodule(HEAD_NAME)
    if installed_embedding.weight is not installed_head.weight:
        raise RuntimeError("candidate installation did not preserve the weight tie")
    metrics = evaluate_domains(model, gates, args.device)
    metric_ratios = ratios(metrics, baseline)
    incremental_ratios = ratios(metrics, parent_metrics)
    passed = all(value <= args.gate_ratio for value in metric_ratios.values()) and all(
        value <= args.incremental_gate_ratio for value in incremental_ratios.values()
    )

    artifact_path = None
    artifact_sha256 = None
    if args.write_artifact and passed:
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    source_ternary_weights = sum(accepted_weight_counts(source).values())
    low_bit_weights = (
        source_ternary_weights
        + install_result.new_q2_weights
        + install_result.q4_weights
        + install_result.q8_weights
    )
    result = {
        "schema": "wal-tat-tied-embedding-low-bit-v1",
        "tag": args.tag,
        "precision": args.precision,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_names": [EMBEDDING_NAME, HEAD_NAME],
        "shared_weight_verified": installed_embedding.weight is installed_head.weight,
        "target_shape": [rows, columns],
        "target_weights_counted_once": rows * columns,
        "moment_sequences": args.moment_sequences,
        "head_moment_weight": args.head_moment_weight,
        "head_moment_statistics": {
            "minimum": float(head_moment.min().item()),
            "mean": float(head_moment.mean().item()),
            "maximum": float(head_moment.max().item()),
        },
        "projection_statistics": projection_statistics,
        "baseline_metrics": baseline,
        "parent_metrics": parent_metrics,
        "candidate_metrics": metrics,
        "ratios_vs_raw_bf16": metric_ratios,
        "ratios_vs_parent": incremental_ratios,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "passed": passed,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "source_ternary_weights": source_ternary_weights,
        "new_q2_weights": install_result.new_q2_weights,
        "q4_weights": install_result.q4_weights,
        "q8_weights": install_result.q8_weights,
        "low_bit_weights": low_bit_weights,
        "low_bit_coverage": low_bit_weights / args.total_major_weights,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else None
        ),
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}; precision={args.precision} passed={passed} "
        f"ratios={metric_ratios} artifact={artifact_path}",
        flush=True,
    )
    del model, source, parent, artifact, matrices, codes, scales
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
