"""Find the minimum signed-Q8 rescue for a mixed Q2/Q4 block stage."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from block_proxy_recovery import collect_input_second_moments
from compensation_window import (
    PROJECT,
    default_model_path,
    evaluate_domains,
    install_checkpoint,
    load_model,
    ratios,
    seed_everything,
    set_submodule,
    sha256_file,
)
from macro_mlp_q4_rescue import FixedWeightLinear, global_rescue_masks, parse_fractions
from wal_tat import (
    install_mixed_q2_q4_artifact,
    q2_g128_physical_bpw,
    q4_g128_physical_bpw,
    q8_g128_physical_bpw,
    weighted_symmetric_q8_project,
)


PROJECTION_MAP = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument(
        "--target-projections",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
    )
    parser.add_argument("--moment-sequences", type=int, default=128)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument(
        "--rescue-fractions", default="0.01,0.02,0.05,0.1,0.2,0.4,1.0"
    )
    parser.add_argument(
        "--rescue-projections",
        help="optional comma-separated subset eligible for Q8 rescue",
    )
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.005)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.rescue_fractions)
    source_path = args.source_checkpoint.resolve()
    parent_path = args.parent_artifact.resolve()
    candidate_path = args.candidate_artifact.resolve()
    suite_path = args.suite.resolve()
    model_path = (args.model_path or default_model_path()).resolve()
    source_hash = sha256_file(source_path)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    parent_artifact = torch.load(parent_path, map_location="cpu", weights_only=False)
    candidate_artifact = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
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
    requested = tuple(
        item.strip() for item in args.target_projections.split(",") if item.strip()
    )
    if not requested or any(item not in PROJECTION_MAP for item in requested):
        raise ValueError("invalid target projections")
    names = tuple(
        f"model.layers.{args.target_layer}.{PROJECTION_MAP[item]}"
        for item in requested
    )
    if any(name not in candidate_artifact["matrices"] for name in names):
        raise ValueError("candidate artifact lacks a target matrix")
    rescue_requested = (
        tuple(
            item.strip()
            for item in args.rescue_projections.split(",")
            if item.strip()
        )
        if args.rescue_projections
        else requested
    )
    if not rescue_requested or any(item not in requested for item in rescue_requested):
        raise ValueError("rescue projections must be a subset of target projections")
    rescue_names = {
        f"model.layers.{args.target_layer}.{PROJECTION_MAP[item]}"
        for item in rescue_requested
    }

    seed_everything(args.seed)
    started = time.time()
    model = load_model(model_path, args.device)
    baseline = evaluate_domains(model, gates, args.device)
    selection_baseline = evaluate_domains(model, selection_gates, args.device)
    install_checkpoint(model, source_payload, args.device)
    install_mixed_q2_q4_artifact(
        model,
        parent_artifact,
        source_payload,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    parent = evaluate_domains(model, gates, args.device)
    parent_selection = evaluate_domains(model, selection_gates, args.device)
    moments = collect_input_second_moments(
        model, names, calibration, args.device, args.moment_sequences
    )
    original = {
        name: model.get_submodule(name).weight.detach().float().clone()
        for name in names
    }
    biases = {name: model.get_submodule(name).bias for name in names}
    install_mixed_q2_q4_artifact(
        model,
        candidate_artifact,
        source_payload,
        device=args.device,
        expected_source_sha256=source_hash,
    )
    candidate = evaluate_domains(model, gates, args.device)

    q2_grouped = {}
    q4_grouped = {}
    q8_grouped = {}
    q8_codes = {}
    q8_scales = {}
    benefit = {}
    eligible = {}
    total_groups = 0
    eligible_groups = 0
    for name in names:
        entry = candidate_artifact["matrices"][name]
        q2_codes = entry["q2_codes_int8"].to(args.device).float()
        q2_scales = entry["q2_scales_fp16"].to(args.device).float()
        q4_codes = entry["q4_codes_int8"].to(args.device).float()
        q4_scales = entry["q4_scales_fp16"].to(args.device).float()
        q4_mask = entry["q4_mask"].to(args.device).bool()
        new_codes, new_scales, q8_error = weighted_symmetric_q8_project(
            original[name], moments[name], group_size=128
        )
        q2_value = q2_codes * q2_scales.unsqueeze(-1)
        q4_value = q4_codes * q4_scales.unsqueeze(-1)
        q8_value = new_codes.float() * new_scales.unsqueeze(-1)
        rows, columns = map(int, entry["shape"])
        grouped_original = torch.nn.functional.pad(
            original[name], (0, (-columns) % 128)
        ).view_as(q4_value)
        moment = moments[name]
        if (-columns) % 128:
            moment = torch.nn.functional.pad(moment, (0, (-columns) % 128))
        grouped_moment = moment.view(1, -1, 128).expand_as(q4_value)
        q4_error = (
            grouped_moment * (grouped_original - q4_value).square()
        ).sum(-1)
        q2_grouped[name] = q2_value
        q4_grouped[name] = q4_value
        q8_grouped[name] = q8_value
        q8_codes[name] = new_codes.cpu()
        q8_scales[name] = new_scales.half().cpu()
        benefit[name] = q4_error - q8_error
        eligible[name] = q4_mask if name in rescue_names else torch.zeros_like(q4_mask)
        total_groups += q4_mask.numel()
        eligible_groups += int(eligible[name].sum().item())

    def install_rescue(mask_map: dict[str, torch.Tensor]) -> None:
        for name in names:
            entry = candidate_artifact["matrices"][name]
            q4_mask = entry["q4_mask"].to(args.device).bool()
            grouped = torch.where(
                q4_mask.unsqueeze(-1), q4_grouped[name], q2_grouped[name]
            )
            grouped = torch.where(
                mask_map[name].unsqueeze(-1), q8_grouped[name], grouped
            )
            rows, columns = map(int, entry["shape"])
            set_submodule(
                model,
                name,
                FixedWeightLinear(
                    grouped.reshape(rows, -1)[:, :columns].to(torch.bfloat16),
                    biases[name],
                ).to(args.device),
            )

    candidates = {}
    candidate_masks = {}
    for fraction in fractions:
        count = min(eligible_groups, max(1, round(eligible_groups * fraction)))
        masks = global_rescue_masks(benefit, eligible, count)
        install_rescue(masks)
        metrics = evaluate_domains(model, selection_gates, args.device)
        metric_ratios = ratios(metrics, selection_baseline)
        incremental = ratios(metrics, parent_selection)
        actual_fraction = count / eligible_groups
        q8_fraction_all_target = count / total_groups
        target_bpw = q4_g128_physical_bpw() + q8_fraction_all_target * (
            q8_g128_physical_bpw() - q4_g128_physical_bpw()
        )
        key = f"q8_{actual_fraction:.8f}"
        candidates[key] = {
            "requested_fraction": fraction,
            "rescued_groups": count,
            "eligible_groups": eligible_groups,
            "eligible_q8_fraction": actual_fraction,
            "all_target_q8_fraction": q8_fraction_all_target,
            "target_physical_bpw": target_bpw,
            "selection_metrics": metrics,
            "selection_ratios": metric_ratios,
            "selection_incremental_ratios_vs_parent": incremental,
            "selection_passed": all(
                value <= args.gate_ratio for value in metric_ratios.values()
            )
            and all(
                value <= args.incremental_gate_ratio
                for value in incremental.values()
            ),
        }
        candidate_masks[key] = {name: mask.cpu() for name, mask in masks.items()}
        print(key, metric_ratios, incremental, flush=True)

    full_evaluations = {}
    first_full_pass = None
    for key in sorted(candidates, key=lambda item: candidates[item]["rescued_groups"]):
        if not candidates[key]["selection_passed"]:
            continue
        masks = {
            name: mask.to(args.device)
            for name, mask in candidate_masks[key].items()
        }
        install_rescue(masks)
        metrics = evaluate_domains(model, gates, args.device)
        metric_ratios = ratios(metrics, baseline)
        incremental = ratios(metrics, parent)
        passed = all(
            value <= args.gate_ratio for value in metric_ratios.values()
        ) and all(
            value <= args.incremental_gate_ratio for value in incremental.values()
        )
        full_evaluations[key] = {
            "metrics": metrics,
            "ratios": metric_ratios,
            "incremental_ratios_vs_parent": incremental,
            "passed": passed,
        }
        print(f"full {key} passed={passed} ratios={metric_ratios}", flush=True)
        if passed:
            first_full_pass = key
            break

    artifact_path = None
    artifact_sha256 = None
    if args.write_artifact and first_full_pass is not None:
        recovered = copy.deepcopy(candidate_artifact)
        recovered["format"] = "wal-tat-mixed-q2-q4-q8-v1"
        recovered["q8_physical_bpw"] = q8_g128_physical_bpw()
        recovered["parent_artifact_sha256"] = sha256_file(parent_path)
        recovered["q8_rescue"] = {
            "source_candidate_sha256": sha256_file(candidate_path),
            "candidate": first_full_pass,
            "development_ratios": full_evaluations[first_full_pass]["ratios"],
            "development_incremental_ratios_vs_parent": full_evaluations[
                first_full_pass
            ]["incremental_ratios_vs_parent"],
        }
        for name, entry in recovered["matrices"].items():
            if name in names:
                q8_mask = candidate_masks[first_full_pass][name].bool()
                entry["q4_mask"] = entry["q4_mask"].bool() & ~q8_mask
                entry["q8_mask"] = q8_mask
                entry["q8_codes_int8"] = q8_codes[name]
                entry["q8_scales_fp16"] = q8_scales[name]
            else:
                entry["q8_mask"] = torch.zeros_like(entry["q4_mask"], dtype=torch.bool)
                entry["q8_codes_int8"] = torch.zeros_like(
                    entry["q4_codes_int8"], dtype=torch.int8
                )
                entry["q8_scales_fp16"] = torch.ones_like(
                    entry["q4_scales_fp16"], dtype=torch.float16
                )
        artifact_path = (
            source_path.parents[1]
            / "artifacts"
            / f"wal-tat-{args.tag}-mixed-q2-q4-q8.pt"
        )
        torch.save(recovered, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)

    result = {
        "schema": "wal-tat-macro-q8-rescue-v1",
        "tag": args.tag,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": sha256_file(parent_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256_file(candidate_path),
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "target_layer": args.target_layer,
        "target_projections": requested,
        "rescue_projections": rescue_requested,
        "moment_sequences": args.moment_sequences,
        "selection_gate_sequences": args.selection_gate_sequences,
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "parent_metrics": parent,
        "parent_ratios": ratios(parent, baseline),
        "candidate_metrics": candidate,
        "candidate_ratios": ratios(candidate, baseline),
        "candidate_incremental_ratios_vs_parent": ratios(candidate, parent),
        "total_target_groups": total_groups,
        "eligible_q4_groups": eligible_groups,
        "candidate_results": candidates,
        "full_evaluations": full_evaluations,
        "first_full_pass": first_full_pass,
        "development_passed": first_full_pass is not None,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_sha256": artifact_sha256,
        "artifact_written": artifact_path is not None,
        "checkpoint_written": False,
        "sealed_audit_opened": False,
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
        f"wrote {output}; first_full_pass={first_full_pass} "
        f"artifact_written={artifact_path is not None}",
        flush=True,
    )
    del model, moments, original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
