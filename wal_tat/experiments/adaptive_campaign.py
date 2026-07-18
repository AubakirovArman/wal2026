"""Run real WAL-TAT transactions through development and independent audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from wal_tat import AdaptiveTransactionSizer
from wal_tat.campaign import (
    accepted_weight_counts,
    atomic_write_json,
    coverage_proportional_nll_gate,
    validate_checkpoint_deletion_target,
    worst_ratio,
)


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CHECKPOINT_DIR = WORKSPACE / "wal2/checkpoints"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_python(script: str, arguments: list[str], visible_device: str) -> list[str]:
    command, environment = python_command(script, arguments, visible_device)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, env=environment, check=True)
    return command


def python_command(
    script: str, arguments: list[str], visible_device: str
) -> tuple[list[str], dict[str, str]]:
    command = [sys.executable, str(PROJECT / f"experiments/{script}"), *arguments]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = visible_device
    environment["PYTHONPATH"] = str(PROJECT / "src")
    environment.setdefault("HF_DATASETS_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    return command, environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--development-suite", type=Path, required=True)
    parser.add_argument("--audit-suite", type=Path, action="append", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--hidden-module", required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--initial-fraction", type=float, default=1 / 256)
    parser.add_argument("--minimum-fraction", type=float, default=1 / 1024)
    parser.add_argument("--maximum-fraction", type=float, default=1 / 64)
    parser.add_argument("--target-coverage", type=float, default=1.0)
    parser.add_argument("--tight-headroom", type=float, default=0.35)
    parser.add_argument("--roomy-headroom", type=float, default=0.75)
    parser.add_argument("--grow-after", type=int, default=2)
    parser.add_argument("--channel-blocks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gate-ratio", type=float, default=1.00195)
    parser.add_argument("--dynamic-coverage-gate", action="store_true")
    parser.add_argument("--candidate-weights", type=int)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--total-major-weights", type=int)
    parser.add_argument("--full-model-nll-budget", type=float, default=0.05)
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.1)
    parser.add_argument("--teacher-source", choices=("frontier", "bf16"), default="bf16")
    parser.add_argument("--arm", default="candidate_bf16_mlp")
    parser.add_argument("--train-visible-device", default="0")
    parser.add_argument("--audit-visible-device", action="append", default=[])
    parser.add_argument("--delete-superseded", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < args.target_coverage <= 1:
        raise ValueError("target coverage must be in (0, 1]")
    if args.dynamic_coverage_gate and (
        not args.candidate_weights or not args.total_major_weights
    ):
        raise ValueError(
            "dynamic coverage gate requires candidate and total major weights"
        )
    source = args.source_checkpoint.expanduser().resolve(strict=True)
    development_suite = args.development_suite.expanduser().resolve(strict=True)
    audit_suites = [path.expanduser().resolve(strict=True) for path in args.audit_suite]
    audit_devices = args.audit_visible_device or [args.train_visible_device]
    state_path = args.state.expanduser().resolve()

    if state_path.exists():
        state = load_json(state_path)
        if state.get("status", "active") != "active":
            raise ValueError(f"campaign state is not active: {state.get('status')}")
        if "gate_policy" not in state and "gate_ratio" in state:
            state["gate_policy"] = {
                "dynamic": False,
                "fixed_gate_ratio": state["gate_ratio"],
                "candidate_weights": None,
                "group_size": 128,
                "total_major_weights": None,
                "full_model_nll_budget": 0.05,
            }
        gate_policy = {
            "dynamic": args.dynamic_coverage_gate,
            "fixed_gate_ratio": args.gate_ratio,
            "candidate_weights": args.candidate_weights,
            "group_size": args.group_size,
            "total_major_weights": args.total_major_weights,
            "full_model_nll_budget": args.full_model_nll_budget,
        }
        expected = {
            "candidate_name": args.candidate_name,
            "development_suite": str(development_suite),
            "audit_suites": [str(path) for path in audit_suites],
            "gate_policy": gate_policy,
        }
        mismatched = {
            key: {"state": state.get(key), "requested": value}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatched:
            raise ValueError(f"campaign resume configuration mismatch: {mismatched}")
        source = Path(state["frontier"]["checkpoint"]).resolve(strict=True)
        initial_fraction = float(state["next_fraction"])
        history = list(state["history"])
    else:
        initial_fraction = args.initial_fraction
        history = []
        gate_policy = {
            "dynamic": args.dynamic_coverage_gate,
            "fixed_gate_ratio": args.gate_ratio,
            "candidate_weights": args.candidate_weights,
            "group_size": args.group_size,
            "total_major_weights": args.total_major_weights,
            "full_model_nll_budget": args.full_model_nll_budget,
        }
        state = {
            "schema": "wal-tat-adaptive-campaign-v1",
            "status": "active",
            "created_ns": time.time_ns(),
            "candidate_name": args.candidate_name,
            "development_suite": str(development_suite),
            "audit_suites": [str(path) for path in audit_suites],
            "gate_policy": gate_policy,
            "frontier": {
                "checkpoint": str(source),
                "sha256": sha256_file(source),
            },
            "next_fraction": initial_fraction,
            "history": history,
        }
        atomic_write_json(state_path, state)

    sizer = AdaptiveTransactionSizer(
        initial_fraction,
        minimum_fraction=args.minimum_fraction,
        maximum_fraction=args.maximum_fraction,
        tight_headroom=args.tight_headroom,
        roomy_headroom=args.roomy_headroom,
        grow_after=args.grow_after,
    )

    for _ in range(args.iterations):
        step = len(history) + 1
        tag = f"{args.tag_prefix}_s{step:04d}"
        fraction = sizer.current_fraction
        effective_gate_ratio = args.gate_ratio
        projected_accepted_weights = None
        if args.dynamic_coverage_gate:
            source_payload = torch.load(source, map_location="cpu", weights_only=False)
            counts = accepted_weight_counts(source_payload)
            source_accepted = sum(counts.values())
            candidate_accepted = counts.get(args.candidate_name, 0)
            requested = round(
                args.candidate_weights / args.group_size * fraction
            ) * args.group_size
            added = min(requested, args.candidate_weights - candidate_accepted)
            projected_accepted_weights = source_accepted + added
            effective_gate_ratio = coverage_proportional_nll_gate(
                projected_accepted_weights,
                args.total_major_weights,
                args.full_model_nll_budget,
            )
        experiment_arguments = [
            "--source-checkpoint", str(source),
            "--suite", str(development_suite),
            "--candidate-name", args.candidate_name,
            "--tag", tag,
            "--fraction", repr(fraction),
            "--channel-blocks", str(args.channel_blocks),
            "--steps", str(args.steps),
            "--lr", repr(args.lr),
            "--gate-ratio", repr(effective_gate_ratio),
            "--kd-weight", repr(args.kd_weight),
            "--hidden-kd-weight", repr(args.hidden_kd_weight),
            "--hidden-module", args.hidden_module,
            "--teacher-source", args.teacher_source,
            "--arms", args.arm,
            "--device", "cuda",
        ]
        command = run_python(
            "compensation_window.py", experiment_arguments, args.train_visible_device
        )
        result_path = PROJECT / f"results/{tag}.json"
        experiment = load_json(result_path)
        development_chosen = experiment.get("chosen_arm")
        record = {
            "step": step,
            "tag": tag,
            "fraction": fraction,
            "source_checkpoint": str(source),
            "experiment_result": str(result_path),
            "experiment_command": command,
            "development_chosen_arm": development_chosen,
            "effective_gate_ratio": effective_gate_ratio,
            "projected_accepted_weights": projected_accepted_weights,
            "arm_audits": {},
        }

        passing_candidates = {
            name: Path(value["checkpoint"]).resolve(strict=True)
            for name, value in experiment["arms"].items()
            if value["passed"]
        }
        if not passing_candidates:
            attempted = [
                max(value["attempted_ratios"].values())
                for value in experiment["arms"].values()
            ]
            observed_worst = max(attempted)
            passed = False
            chosen = None
        else:
            candidate_digests = {
                name: sha256_file(path) for name, path in passing_candidates.items()
            }
            audit_documents = {name: [] for name in passing_candidates}
            digest_matches = {name: True for name in passing_candidates}
            audit_specs = []
            for arm_name, candidate_path in passing_candidates.items():
                record["arm_audits"][arm_name] = []
                for index, suite in enumerate(audit_suites, start=1):
                    audit_tag = f"{tag}_{arm_name}_audit_{index}"
                    audit_arguments = [
                        str(candidate_path),
                        str(suite),
                        "--tag", audit_tag,
                        "--gate-ratio", repr(effective_gate_ratio),
                        "--device", "cuda",
                    ]
                    audit_path = PROJECT / f"results/{audit_tag}.json"
                    audit_specs.append(
                        (arm_name, suite, audit_path, audit_arguments)
                    )
            # Run at most one verifier per declared GPU at a time. Additional
            # arms form later waves instead of oversubscribing the same GPU.
            for start in range(0, len(audit_specs), len(audit_devices)):
                wave = []
                for device_index, spec in enumerate(
                    audit_specs[start : start + len(audit_devices)]
                ):
                    arm_name, suite, audit_path, audit_arguments = spec
                    audit_command, audit_environment = python_command(
                        "verify_checkpoint.py",
                        audit_arguments,
                        audit_devices[device_index],
                    )
                    print("+", " ".join(audit_command), flush=True)
                    process = subprocess.Popen(
                        audit_command, cwd=PROJECT, env=audit_environment
                    )
                    wave.append(
                        (process, arm_name, suite, audit_path, audit_command)
                    )
                for process, arm_name, suite, audit_path, audit_command in wave:
                    return_code = process.wait()
                    if return_code:
                        raise subprocess.CalledProcessError(return_code, audit_command)
                    audit = load_json(audit_path)
                    digest_matches[arm_name] &= (
                        audit.get("checkpoint_sha256") == candidate_digests[arm_name]
                    )
                    audit_documents[arm_name].append(audit)
                    record["arm_audits"][arm_name].append(
                        {
                            "suite": str(suite),
                            "result": str(audit_path),
                            "command": audit_command,
                            "passed": bool(audit["passed"]),
                            "ratios": audit["ratios"],
                        }
                    )
            arm_worst = {
                name: worst_ratio(documents)
                for name, documents in audit_documents.items()
            }
            valid_arms = [
                name
                for name, documents in audit_documents.items()
                if digest_matches[name]
                and all(bool(document["passed"]) for document in documents)
            ]
            chosen = min(valid_arms, key=arm_worst.get) if valid_arms else None
            passed = chosen is not None
            observed_worst = (
                arm_worst[chosen] if passed else min(arm_worst.values())
            )
            record["holdout_worst_by_arm"] = arm_worst
            record["holdout_chosen_arm"] = chosen

        decision = sizer.observe(
            passed=passed,
            worst_ratio=observed_worst,
            gate_ratio=effective_gate_ratio,
        )
        record["passed"] = passed
        record["worst_ratio"] = observed_worst
        record["size_decision"] = decision.__dict__

        cleanup_targets: list[tuple[str, Path]] = []
        if passed:
            previous = source
            candidate = passing_candidates[chosen]
            source = candidate
            coverage = float(
                experiment["arms"][chosen]["coverage"][args.candidate_name]
            )
            record["accepted_checkpoint"] = str(source)
            record["accepted_checkpoint_sha256"] = sha256_file(source)
            record["candidate_coverage"] = coverage
            state["frontier"] = {
                "checkpoint": str(source),
                "sha256": record["accepted_checkpoint_sha256"],
                "candidate_coverage": coverage,
            }
            if args.delete_superseded and previous != source:
                cleanup_targets.append(("superseded", previous))
            if args.delete_superseded:
                cleanup_targets.extend(
                    ("unchosen_arm", path)
                    for name, path in passing_candidates.items()
                    if name != chosen
                )
        elif passing_candidates:
            record["rejected_checkpoints"] = [
                str(path) for path in passing_candidates.values()
            ]
            if args.delete_superseded:
                cleanup_targets.extend(
                    ("rejected", path) for path in passing_candidates.values()
                )

        history.append(record)
        state["history"] = history
        state["next_fraction"] = sizer.current_fraction
        state["updated_ns"] = time.time_ns()
        # Publish the new accepted frontier before removing its predecessor.
        # A crash can leave an extra file, but never a state file that points
        # only to an already deleted checkpoint.
        atomic_write_json(state_path, state)
        if cleanup_targets:
            record["deleted_checkpoints"] = []
            seen = set()
            for reason, cleanup_target in cleanup_targets:
                deletion = validate_checkpoint_deletion_target(
                    cleanup_target, CHECKPOINT_DIR
                )
                if deletion in seen:
                    continue
                seen.add(deletion)
                deleted_bytes = deletion.stat().st_size
                deletion.unlink()
                record["deleted_checkpoints"].append(
                    {
                        "reason": reason,
                        "path": str(deletion),
                        "bytes": deleted_bytes,
                        "recoverable_without_rerun": False,
                    }
                )
            state["updated_ns"] = time.time_ns()
            atomic_write_json(state_path, state)
        print(
            f"campaign step={step} passed={passed} worst={observed_worst:.9f} "
            f"next_fraction={sizer.current_fraction:.9f}",
            flush=True,
        )
        if passed and record["candidate_coverage"] >= args.target_coverage:
            break


if __name__ == "__main__":
    main()
