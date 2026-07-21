"""Move the safest fraction of a tied embedding/head from Q8 to Q4."""
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
from reverse_q4_to_q2_fraction import parse_fractions
from tied_embedding_low_bit import (
    EMBEDDING_NAME,
    GROUP_SIZE,
    HEAD_NAME,
    collect_head_input_moment,
    grouped_codes,
    project_rows,
)
from wal_tat import install_mixed_q2_q4_artifact, valid_group_weight_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--fractions", default="0.01,0.025,0.05,0.1,0.25,0.5,0.75,0.9,1"
    )
    parser.add_argument("--moment-sequences", type=int, default=64)
    parser.add_argument("--selection-gate-sequences", type=int, default=16)
    parser.add_argument("--projection-row-chunk", type=int, default=1024)
    parser.add_argument("--head-moment-weight", type=float, default=0.8)
    parser.add_argument(
        "--input-frequency-weight",
        type=float,
        default=0.25,
        help="protect rows used frequently as input embeddings during selection",
    )
    parser.add_argument("--gate-ratio", type=float, default=1.1)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.1)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=887)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def candidate_with_q4_fraction(
    parent: dict,
    selected: torch.Tensor,
    q4_codes: torch.Tensor,
    q4_scales: torch.Tensor,
) -> dict:
    candidate = dict(parent)
    matrices = dict(parent["matrices"])
    entry = dict(matrices[EMBEDDING_NAME])
    parent_q8 = entry["q8_mask"].bool()
    if selected.shape != parent_q8.shape or torch.logical_and(selected, ~parent_q8).any():
        raise ValueError("tied Q4 selection is outside the parent Q8 mask")
    entry["q4_codes_int8"] = q4_codes
    entry["q4_scales_fp16"] = q4_scales
    entry["q4_mask"] = entry["q4_mask"].bool() | selected
    entry["q8_mask"] = parent_q8 & ~selected
    matrices[EMBEDDING_NAME] = entry
    candidate["matrices"] = matrices
    return candidate


def selection_mask_from_order(
    eligible_flat: torch.Tensor,
    order: torch.Tensor,
    shape: tuple[int, int],
    count: int,
) -> torch.Tensor:
    result = torch.zeros(shape[0] * shape[1], dtype=torch.bool)
    result[eligible_flat.index_select(0, order[:count])] = True
    return result.view(shape)


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.fractions)
    if not 0 <= args.head_moment_weight <= 1:
        raise ValueError("head moment weight must be in [0, 1]")
    if args.input_frequency_weight < 0:
        raise ValueError("input frequency weight must be non-negative")
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    parent_artifact = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent_artifact.get("format") != "wal-tat-mixed-q2-q4-q8-v1":
        raise ValueError("parent artifact must use the mixed Q2/Q4/Q8 format")
    entry = parent_artifact["matrices"].get(EMBEDDING_NAME)
    if entry is None or entry.get("kind") != "tied_embedding_head":
        raise ValueError("parent artifact lacks a tied embedding/head entry")
    if entry.get("tied_linear_name") != HEAD_NAME:
        raise ValueError("parent artifact has an unexpected tied output head")
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    calibration = suite["calibration"]
    gates = suite["gates"]
    selection_gates = {
        domain: chunks[: args.selection_gate_sequences]
        for domain, chunks in gates.items()
    }
    if any(
        len(chunks) != args.selection_gate_sequences
        for chunks in selection_gates.values()
    ):
        raise ValueError("suite lacks requested selection sequences")

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    embedding = model.get_submodule(EMBEDDING_NAME)
    head = model.get_submodule(HEAD_NAME)
    if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
        raise TypeError("expected tied embedding/head modules")
    if embedding.weight is not head.weight:
        raise ValueError("raw model embedding and head are not tied")
    raw_weight = embedding.weight.detach().clone()
    rows, columns = map(int, raw_weight.shape)
    groups = columns // GROUP_SIZE
    shape = (rows, groups)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    install_checkpoint(model, source, args.device)
    install_mixed_q2_q4_artifact(
        model,
        parent_artifact,
        source,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    parent = evaluate_domains(model, gates, args.device)
    parent_selection = evaluate_domains(model, selection_gates, args.device)
    head_moment = collect_head_input_moment(
        model,
        calibration,
        device=args.device,
        count=args.moment_sequences,
    )
    normalized_head = head_moment / head_moment.mean().clamp_min(1e-12)
    moment = torch.lerp(
        torch.ones_like(normalized_head), normalized_head, args.head_moment_weight
    )
    q4_codes, q4_scales, projection_statistics = project_rows(
        raw_weight,
        moment,
        precision="q4",
        row_chunk=args.projection_row_chunk,
    )
    q4_codes = grouped_codes(q4_codes, rows=rows, columns=columns)

    grouped_raw = raw_weight.float().view(rows, groups, GROUP_SIZE)
    grouped_moment = moment.view(1, groups, GROUP_SIZE)
    q4_value = q4_codes.to(args.device).float() * q4_scales.to(args.device).float().unsqueeze(-1)
    q8_codes = entry["q8_codes_int8"].to(args.device)
    q8_scales = entry["q8_scales_fp16"].to(args.device).float()
    q8_value = q8_codes.float() * q8_scales.unsqueeze(-1)
    q4_error = ((grouped_raw - q4_value).square() * grouped_moment).sum(-1)
    q8_error = ((grouped_raw - q8_value).square() * grouped_moment).sum(-1)

    token_counts = torch.zeros(rows, dtype=torch.float64)
    for chunk in calibration[: args.moment_sequences]:
        token_counts += torch.bincount(chunk.reshape(-1), minlength=rows).double()
    mean_nonzero = token_counts[token_counts > 0].mean().clamp_min(1.0)
    frequency_factor = 1.0 + args.input_frequency_weight * torch.sqrt(
        token_counts / mean_nonzero
    )
    damage = (q4_error - q8_error).cpu() * frequency_factor.float().unsqueeze(-1)
    eligible = entry["q8_mask"].bool().cpu()
    eligible_flat = torch.where(eligible.reshape(-1))[0]
    if eligible_flat.numel() == 0:
        raise ValueError("tied matrix contains no Q8 groups")
    eligible_damage = damage.reshape(-1).index_select(0, eligible_flat)
    order = torch.argsort(eligible_damage, stable=True)
    del q4_value, q8_value, q8_codes, q8_scales, q4_error, q8_error
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attempts = []
    passing = []
    for fraction in fractions:
        count = max(1, min(eligible_flat.numel(), round(eligible_flat.numel() * fraction)))
        selected = selection_mask_from_order(eligible_flat, order, shape, count)
        candidate = candidate_with_q4_fraction(
            parent_artifact, selected, q4_codes, q4_scales
        )
        install_mixed_q2_q4_artifact(
            model,
            candidate,
            source,
            device=args.device,
            expected_source_sha256=source_hash,
        )
        metrics = evaluate_domains(model, selection_gates, args.device)
        absolute = ratios(metrics, selection_baseline)
        incremental = ratios(metrics, parent_selection)
        passed = all(value <= args.gate_ratio for value in absolute.values()) and all(
            value <= args.incremental_gate_ratio for value in incremental.values()
        )
        selected_weights = valid_group_weight_count(selected, columns, GROUP_SIZE)
        attempt = {
            "fraction": fraction,
            "selected_groups": count,
            "selected_weights": selected_weights,
            "selection_metrics": metrics,
            "selection_ratios": absolute,
            "selection_incremental_ratios_vs_parent": incremental,
            "passed": passed,
        }
        attempts.append(attempt)
        print(
            f"fraction={fraction:.6f} groups={count} passed={passed} "
            f"ratios={absolute} incremental={incremental}",
            flush=True,
        )
        if passed:
            passing.append((count, fraction, selected, attempt))

    accepted = None
    full_evaluations = []
    final_candidate = None
    for count, fraction, selected, attempt in sorted(passing, reverse=True, key=lambda item: item[0]):
        candidate = candidate_with_q4_fraction(
            parent_artifact, selected, q4_codes, q4_scales
        )
        install_mixed_q2_q4_artifact(
            model,
            candidate,
            source,
            device=args.device,
            expected_source_sha256=source_hash,
        )
        metrics = evaluate_domains(model, gates, args.device)
        absolute = ratios(metrics, baseline)
        incremental = ratios(metrics, parent)
        passed = all(value <= args.gate_ratio for value in absolute.values()) and all(
            value <= args.incremental_gate_ratio for value in incremental.values()
        )
        evaluation = {
            "fraction": fraction,
            "selected_groups": count,
            "selected_weights": attempt["selected_weights"],
            "metrics": metrics,
            "ratios": absolute,
            "incremental_ratios_vs_parent": incremental,
            "passed": passed,
        }
        full_evaluations.append(evaluation)
        print(
            f"full fraction={fraction:.6f} passed={passed} "
            f"ratios={absolute} incremental={incremental}",
            flush=True,
        )
        if passed:
            accepted = evaluation
            final_candidate = candidate
            break

    artifact_path = None
    artifact_sha256 = None
    if accepted is not None and args.write_artifact:
        final_candidate["parent_artifact_sha256"] = sha256_file(parent_path)
        final_candidate["reverse_tied_q8_to_q4"] = accepted
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
        )
        torch.save(final_candidate, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-reverse-tied-q8-to-q4-fraction-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "fractions": fractions,
        "moment_sequences": args.moment_sequences,
        "selection_gate_sequences": args.selection_gate_sequences,
        "head_moment_weight": args.head_moment_weight,
        "input_frequency_weight": args.input_frequency_weight,
        "projection_statistics": projection_statistics,
        "eligible_groups": int(eligible_flat.numel()),
        "eligible_weights": valid_group_weight_count(eligible, columns, GROUP_SIZE),
        "baseline": baseline,
        "parent": parent,
        "parent_ratios": ratios(parent, baseline),
        "attempts": attempts,
        "full_evaluations": full_evaluations,
        "accepted": accepted,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "passed": accepted is not None,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
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
        f"wrote {output}; passed={result['passed']} accepted={accepted} "
        f"artifact={artifact_path}",
        flush=True,
    )
    del model, source, parent_artifact, raw_weight, q4_codes, q4_scales
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
