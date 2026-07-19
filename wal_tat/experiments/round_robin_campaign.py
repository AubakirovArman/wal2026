"""Run several adaptive WAL-TAT campaigns sequentially in one terminal session."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from wal_tat import (
    active_campaign_workers,
    atomic_write_json,
    common_campaign_frontier,
    load_campaign_state,
    synchronize_campaign_frontiers,
    validate_checkpoint_deletion_target,
)


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CHECKPOINT_DIR = WORKSPACE / "wal2/checkpoints"
STEP_SUFFIX = re.compile(r"_s\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--hidden-module", required=True)
    parser.add_argument("--minimum-fraction", type=float, default=1 / 1024)
    parser.add_argument("--maximum-fraction", type=float, default=1 / 64)
    parser.add_argument("--target-coverage", type=float, default=1.0)
    parser.add_argument("--tight-headroom", type=float, default=0.35)
    parser.add_argument("--roomy-headroom", type=float, default=0.75)
    parser.add_argument("--grow-after", type=int, default=2)
    parser.add_argument("--channel-blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--hidden-kd-weight", type=float, default=0.1)
    parser.add_argument("--teacher-source", choices=("frontier", "bf16"), default="bf16")
    parser.add_argument("--arm", default="candidate_only,candidate_bf16_mlp")
    parser.add_argument("--train-visible-device", default="0")
    parser.add_argument("--audit-visible-device", action="append", default=[])
    parser.add_argument("--continue-after-reject", action="store_true")
    return parser.parse_args()


def tag_prefix(state: dict) -> str:
    history = state.get("history", [])
    if not history:
        raise ValueError("cannot infer tag prefix from an empty campaign")
    tag = str(history[-1]["tag"])
    prefix = STEP_SUFFIX.sub("", tag)
    if prefix == tag:
        raise ValueError(f"campaign tag has no numeric step suffix: {tag}")
    return prefix


def adaptive_command(
    state_path: Path, source: Path, args: argparse.Namespace
) -> list[str]:
    state = load_campaign_state(state_path)
    policy = state["gate_policy"]
    command = [
        sys.executable,
        str(PROJECT / "experiments/adaptive_campaign.py"),
        "--source-checkpoint", str(source),
        "--development-suite", state["development_suite"],
    ]
    for suite in state["audit_suites"]:
        command.extend(("--audit-suite", suite))
    command.extend(
        [
            "--candidate-name", state["candidate_name"],
            "--hidden-module", args.hidden_module,
            "--tag-prefix", tag_prefix(state),
            "--state", str(state_path),
            "--iterations", "1",
            "--initial-fraction", repr(float(state["next_fraction"])),
            "--minimum-fraction", repr(args.minimum_fraction),
            "--maximum-fraction", repr(args.maximum_fraction),
            "--target-coverage", repr(args.target_coverage),
            "--tight-headroom", repr(args.tight_headroom),
            "--roomy-headroom", repr(args.roomy_headroom),
            "--grow-after", str(args.grow_after),
            "--channel-blocks", str(args.channel_blocks),
            "--steps", str(args.steps),
            "--lr", repr(args.lr),
            "--gate-ratio", repr(float(policy["fixed_gate_ratio"])),
            "--kd-weight", repr(args.kd_weight),
            "--hidden-kd-weight", repr(args.hidden_kd_weight),
            "--teacher-source", args.teacher_source,
            "--arm", args.arm,
            "--train-visible-device", args.train_visible_device,
        ]
    )
    for device in args.audit_visible_device or [args.train_visible_device]:
        command.extend(("--audit-visible-device", device))
    if policy.get("dynamic"):
        command.extend(
            [
                "--dynamic-coverage-gate",
                "--candidate-weights", str(policy["candidate_weights"]),
                "--group-size", str(policy["group_size"]),
                "--total-major-weights", str(policy["total_major_weights"]),
                "--full-model-nll-budget", repr(float(policy["full_model_nll_budget"])),
            ]
        )
    return command


def generated_checkpoints(record: dict) -> list[Path]:
    experiment = json.loads(Path(record["experiment_result"]).read_text(encoding="utf-8"))
    paths = []
    for arm in experiment.get("arms", {}).values():
        checkpoint = arm.get("checkpoint")
        if checkpoint:
            path = Path(checkpoint)
            if path.exists():
                paths.append(path)
    return paths


def cleanup_step(
    state_path: Path,
    previous: Path,
    accepted: Path,
    all_state_paths: list[Path],
) -> list[dict]:
    """Delete superseded artifacts only after every state references accepted."""
    common, _ = common_campaign_frontier(all_state_paths)
    if common != accepted.resolve():
        raise RuntimeError("refusing cleanup before all campaign frontiers are synchronized")
    state = load_campaign_state(state_path)
    record = state["history"][-1]
    targets: list[tuple[str, Path]] = []
    if previous.resolve() != accepted.resolve() and previous.exists():
        targets.append(("superseded", previous))
    targets.extend(
        ("unchosen_or_rejected", path)
        for path in generated_checkpoints(record)
        if path.resolve() != accepted.resolve()
    )
    deleted = []
    seen: set[Path] = set()
    for reason, target in targets:
        resolved = validate_checkpoint_deletion_target(target, CHECKPOINT_DIR)
        if resolved in seen:
            continue
        seen.add(resolved)
        size = resolved.stat().st_size
        resolved.unlink()
        deleted.append(
            {
                "reason": reason,
                "path": str(resolved),
                "bytes": size,
                "recoverable_without_rerun": False,
            }
        )
    record["deleted_checkpoints"] = deleted
    state["updated_ns"] = time.time_ns()
    atomic_write_json(state_path, state)
    return deleted


def main() -> None:
    args = parse_args()
    if args.cycles < 1:
        raise ValueError("cycles must be positive")
    state_paths = [path.expanduser().resolve(strict=True) for path in args.state]
    summary_path = args.summary.expanduser().resolve()
    workers = active_campaign_workers(workspace=WORKSPACE)
    if workers:
        raise RuntimeError(f"other WAL-TAT workers are already active: {workers}")
    frontier, digest = common_campaign_frontier(state_paths)
    summary = {
        "schema": "wal-tat-round-robin-v1",
        "started_ns": time.time_ns(),
        "states": [str(path) for path in state_paths],
        "initial_frontier": {"checkpoint": str(frontier), "sha256": digest},
        "maximum_concurrent_campaigns": 1,
        "steps": [],
        "status": "active",
    }
    atomic_write_json(summary_path, summary)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src")
    environment.setdefault("HF_DATASETS_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        for cycle in range(1, args.cycles + 1):
            for state_path in state_paths:
                state = load_campaign_state(state_path)
                if float(state["frontier"].get("candidate_coverage") or 0.0) >= args.target_coverage:
                    continue
                previous, _ = common_campaign_frontier(state_paths)
                command = adaptive_command(state_path, previous, args)
                print("+", " ".join(command), flush=True)
                subprocess.run(command, cwd=PROJECT, env=environment, check=True)
                updated = load_campaign_state(state_path)
                record = updated["history"][-1]
                passed = bool(record["passed"])
                accepted = Path(updated["frontier"]["checkpoint"]).resolve(strict=True)
                accepted_digest = str(updated["frontier"]["sha256"])
                synchronize_campaign_frontiers(
                    state_paths, accepted, accepted_digest
                )
                deleted = cleanup_step(
                    state_path, previous, accepted, state_paths
                )
                summary["steps"].append(
                    {
                        "cycle": cycle,
                        "candidate_name": updated["candidate_name"],
                        "campaign_step": record["step"],
                        "passed": passed,
                        "worst_ratio": record["worst_ratio"],
                        "checkpoint": str(accepted),
                        "sha256": accepted_digest,
                        "deleted_checkpoints": deleted,
                    }
                )
                summary["current_frontier"] = {
                    "checkpoint": str(accepted),
                    "sha256": accepted_digest,
                }
                summary["updated_ns"] = time.time_ns()
                atomic_write_json(summary_path, summary)
                if not passed and not args.continue_after_reject:
                    summary["status"] = "stopped_on_reject"
                    summary["completed_ns"] = time.time_ns()
                    atomic_write_json(summary_path, summary)
                    return
        summary["status"] = "completed"
        summary["completed_ns"] = time.time_ns()
        atomic_write_json(summary_path, summary)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["completed_ns"] = time.time_ns()
        atomic_write_json(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
