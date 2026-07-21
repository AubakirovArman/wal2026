"""Sequential relaxed full-model low-bit campaign with crash-safe resume.

The orchestrator never runs more than one GPU child at a time.  Each decoder
block is attempted as strict Q2 first, then full Q4, then full Q8.  A candidate
only becomes the parent of the next stage after a fresh-process verifier passes
all cumulative and incremental NLL-ratio gates.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from compensation_window import PROJECT, WORKSPACE, sha256_file


ALL_PROJECTIONS = "q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--starting-parent", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--start-layer", type=int, default=18)
    parser.add_argument("--end-layer", type=int, default=0)
    parser.add_argument("--tag-prefix", default="relaxed110")
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT / "results/relaxed110_full_model_campaign_state.json",
    )
    parser.add_argument("--gate-ratio", type=float, default=1.1)
    parser.add_argument("--incremental-gate-ratio", type=float, default=1.1)
    parser.add_argument("--moment-sequences", type=int, default=64)
    parser.add_argument("--selection-gate-sequences", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def require_ratio(name: str, value: float) -> None:
    if value < 1.0:
        raise ValueError(f"{name} must be at least 1.0")


def run_child(command: list[str], *, label: str) -> None:
    print(f"campaign child start label={label} command={' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    print(
        f"campaign child end label={label} returncode={completed.returncode}",
        flush=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"child failed for {label}: {completed.returncode}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generation_command(
    args: argparse.Namespace,
    *,
    parent: Path,
    layer: int,
    tag: str,
    precision: str,
) -> list[str]:
    if precision not in {"q2", "q4"}:
        raise ValueError(f"unsupported projection precision: {precision}")
    fraction = "0" if precision == "q2" else "1"
    command = [
        sys.executable,
        str(PROJECT / "experiments/macro_mlp_q4_rescue.py"),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--parent-artifact",
        str(parent),
        "--suite",
        str(args.suite),
        "--tag",
        tag,
        "--target-layer",
        str(layer),
        "--target-projections",
        ALL_PROJECTIONS,
        "--moment-sequences",
        str(args.moment_sequences),
        "--selection-gate-sequences",
        str(args.selection_gate_sequences),
        "--rescue-fractions",
        fraction,
        "--gate-ratio",
        str(args.gate_ratio),
        "--incremental-gate-ratio",
        str(args.incremental_gate_ratio),
        "--write-artifact",
        "--device",
        args.device,
    ]
    if precision == "q2":
        command.append("--allow-zero-rescue")
    return command


def q8_command(
    args: argparse.Namespace,
    *,
    parent: Path,
    q4_candidate: Path,
    layer: int,
    tag: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT / "experiments/macro_q8_rescue.py"),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--parent-artifact",
        str(parent),
        "--candidate-artifact",
        str(q4_candidate),
        "--suite",
        str(args.suite),
        "--tag",
        tag,
        "--target-layer",
        str(layer),
        "--target-projections",
        ALL_PROJECTIONS,
        "--moment-sequences",
        str(args.moment_sequences),
        "--selection-gate-sequences",
        str(args.selection_gate_sequences),
        "--rescue-fractions",
        "1",
        "--gate-ratio",
        str(args.gate_ratio),
        "--incremental-gate-ratio",
        str(args.incremental_gate_ratio),
        "--write-artifact",
        "--device",
        args.device,
    ]


def verify_command(
    args: argparse.Namespace,
    *,
    parent: Path,
    artifact: Path,
    tag: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT / "experiments/verify_mixed_q2_q4_artifact.py"),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--parent-artifact",
        str(parent),
        "--artifact",
        str(artifact),
        "--suite",
        str(args.suite),
        "--tag",
        tag,
        "--gate-ratio",
        str(args.gate_ratio),
        "--incremental-gate-ratio",
        str(args.incremental_gate_ratio),
        "--device",
        args.device,
    ]


def generate(
    args: argparse.Namespace,
    *,
    parent: Path,
    layer: int,
    precision: str,
) -> tuple[Path | None, Path]:
    tag = f"{args.tag_prefix}_block{layer}_{precision}first_v1"
    result_path = PROJECT / f"results/{tag}.json"
    if not result_path.exists():
        run_child(
            generation_command(
                args,
                parent=parent,
                layer=layer,
                tag=tag,
                precision=precision,
            ),
            label=f"layer{layer}-{precision}-generation",
        )
    result = load_json(result_path)
    artifact_value = result.get("artifact")
    if not result.get("development_passed") or not artifact_value:
        return None, result_path
    artifact = Path(artifact_value).resolve()
    if not artifact.exists() or sha256_file(artifact) != result.get("artifact_sha256"):
        raise RuntimeError(f"artifact integrity failure for layer {layer} {precision}")
    return artifact, result_path


def verify(
    args: argparse.Namespace,
    *,
    parent: Path,
    artifact: Path,
    layer: int,
    precision: str,
) -> tuple[bool, Path, dict]:
    tag = f"{args.tag_prefix}_block{layer}_{precision}first_v1_fresh"
    result_path = PROJECT / f"results/{tag}.json"
    if not result_path.exists():
        run_child(
            verify_command(args, parent=parent, artifact=artifact, tag=tag),
            label=f"layer{layer}-{precision}-fresh",
        )
    result = load_json(result_path)
    if result.get("artifact_sha256") != sha256_file(artifact):
        raise RuntimeError(f"verifier artifact mismatch for layer {layer} {precision}")
    if result.get("parent_artifact_sha256") != sha256_file(parent):
        raise RuntimeError(f"verifier parent mismatch for layer {layer} {precision}")
    return bool(result.get("passed")), result_path, result


def generate_q8(
    args: argparse.Namespace,
    *,
    parent: Path,
    q4_candidate: Path,
    layer: int,
) -> tuple[Path | None, Path]:
    tag = f"{args.tag_prefix}_block{layer}_q8fallback_v1"
    result_path = PROJECT / f"results/{tag}.json"
    if not result_path.exists():
        run_child(
            q8_command(
                args,
                parent=parent,
                q4_candidate=q4_candidate,
                layer=layer,
                tag=tag,
            ),
            label=f"layer{layer}-q8-generation",
        )
    result = load_json(result_path)
    artifact_value = result.get("artifact")
    if not result.get("development_passed") or not artifact_value:
        return None, result_path
    artifact = Path(artifact_value).resolve()
    if not artifact.exists() or sha256_file(artifact) != result.get("artifact_sha256"):
        raise RuntimeError(f"Q8 artifact integrity failure for layer {layer}")
    return artifact, result_path


def initial_state(args: argparse.Namespace) -> dict:
    return {
        "schema": "wal-tat-relaxed-full-model-campaign-state-v1",
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "starting_parent": str(args.starting_parent),
        "starting_parent_sha256": sha256_file(args.starting_parent),
        "suite": str(args.suite),
        "suite_sha256": sha256_file(args.suite),
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
        "maximum_concurrent_gpu_children": 1,
        "completed_layers": [],
        "stages": [],
    }


def validate_state(args: argparse.Namespace, state: dict) -> None:
    expected = {
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "starting_parent_sha256": sha256_file(args.starting_parent),
        "suite_sha256": sha256_file(args.suite),
        "gate_ratio": args.gate_ratio,
        "incremental_gate_ratio": args.incremental_gate_ratio,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"campaign state mismatch for {key}")


def main() -> None:
    args = parse_args()
    require_ratio("gate ratio", args.gate_ratio)
    require_ratio("incremental gate ratio", args.incremental_gate_ratio)
    if not 0 <= args.end_layer <= args.start_layer <= 27:
        raise ValueError("layer range must descend within [0, 27]")
    args.source_checkpoint = args.source_checkpoint.resolve()
    args.starting_parent = args.starting_parent.resolve()
    args.suite = args.suite.resolve()
    args.state = args.state.resolve()
    state = load_json(args.state) if args.state.exists() else initial_state(args)
    validate_state(args, state)
    completed = {int(value) for value in state["completed_layers"]}
    parent = args.starting_parent
    for stage in state["stages"]:
        artifact = Path(stage["artifact"]).resolve()
        if sha256_file(artifact) != stage["artifact_sha256"]:
            raise RuntimeError(f"saved stage artifact changed: {artifact}")
        parent = artifact

    for layer in range(args.start_layer, args.end_layer - 1, -1):
        if layer in completed:
            continue
        accepted = None
        accepted_precision = None
        generation_result = None
        verification_result = None
        q4_candidate = None
        for precision in ("q2", "q4"):
            candidate, generation_path = generate(
                args, parent=parent, layer=layer, precision=precision
            )
            if candidate is None:
                continue
            if precision == "q4":
                q4_candidate = candidate
            passed, verify_path, verify_result = verify(
                args,
                parent=parent,
                artifact=candidate,
                layer=layer,
                precision=precision,
            )
            if passed:
                accepted = candidate
                accepted_precision = precision
                generation_result = generation_path
                verification_result = verify_path
                break
            print(
                f"layer {layer} {precision} failed fresh cumulative gate: "
                f"{verify_result.get('incremental_ratios_vs_source')}",
                flush=True,
            )
        if accepted is None:
            if q4_candidate is None:
                raise RuntimeError(f"layer {layer} did not produce a Q4 fallback")
            candidate, generation_path = generate_q8(
                args, parent=parent, q4_candidate=q4_candidate, layer=layer
            )
            if candidate is not None:
                passed, verify_path, verify_result = verify(
                    args,
                    parent=parent,
                    artifact=candidate,
                    layer=layer,
                    precision="q8",
                )
                if passed:
                    accepted = candidate
                    accepted_precision = "q8"
                    generation_result = generation_path
                    verification_result = verify_path
            if accepted is None:
                raise RuntimeError(f"layer {layer} failed even the full-Q8 fallback")

        verify_payload = load_json(verification_result)
        stage = {
            "layer": layer,
            "precision": accepted_precision,
            "parent": str(parent),
            "parent_sha256": sha256_file(parent),
            "artifact": str(accepted),
            "artifact_sha256": sha256_file(accepted),
            "generation_result": str(generation_result),
            "verification_result": str(verification_result),
            "ratios_vs_raw_bf16": verify_payload["ratios"],
            "ratios_vs_strict_source": verify_payload[
                "incremental_ratios_vs_source"
            ],
            "ratios_vs_parent": verify_payload["incremental_ratios_vs_parent"],
            "strict_ternary_weights": verify_payload["strict_ternary_weights"],
            "q4_weights": verify_payload["q4_weights"],
            "q8_weights": verify_payload["q8_weights"],
            "low_bit_weights": verify_payload["low_bit_weights"],
            "low_bit_coverage": verify_payload["low_bit_coverage"],
        }
        state["stages"].append(stage)
        state["completed_layers"].append(layer)
        save_state(args.state, state)
        completed.add(layer)
        parent = accepted
        print(
            f"campaign accepted layer={layer} precision={accepted_precision} "
            f"coverage={stage['low_bit_coverage']:.9f} sha256={stage['artifact_sha256']}",
            flush=True,
        )
    print(
        f"campaign decoder range complete start={args.start_layer} "
        f"end={args.end_layer} parent={parent}",
        flush=True,
    )


if __name__ == "__main__":
    main()
