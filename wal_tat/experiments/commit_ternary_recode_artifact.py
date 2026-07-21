"""Atomically publish an audited coverage-neutral strict ternary recode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from compensation_window import PROJECT, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def grouped(codes: torch.Tensor, shape: tuple[int, int], group_size: int) -> torch.Tensor:
    padding = (-shape[1]) % group_size
    value = F.pad(codes, (0, padding)) if padding else codes
    return value.view(shape[0], -1, group_size)


def atomic_torch_save(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f"refusing to overwrite existing checkpoint: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("development_result", type=Path)
    parser.add_argument("audit_result", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    artifact_path = args.artifact.resolve()
    development_path = args.development_result.resolve()
    audit_path = args.audit_result.resolve()
    policy_path = args.policy.resolve()
    output = args.output.resolve()
    source_hash = sha256_file(checkpoint)
    artifact_hash = sha256_file(artifact_path)
    development_hash = sha256_file(development_path)
    audit_hash = sha256_file(audit_path)
    policy_hash = sha256_file(policy_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    development = load_json(development_path)
    audit = load_json(audit_path)
    require(payload.get("format") == "wal-tat-multi-g128-v2", "bad checkpoint")
    require(
        artifact.get("format") == "wal-tat-ternary-recode-v1", "bad artifact"
    )
    require(
        artifact.get("source_checkpoint_sha256") == source_hash,
        "artifact source mismatch",
    )
    require(development.get("passed") is True, "development result did not pass")
    require(
        development.get("artifact_sha256") == artifact_hash,
        "development artifact mismatch",
    )
    require(
        audit.get("schema") == "wal-tat-ternary-recode-artifact-audit-v1",
        "bad audit schema",
    )
    require(audit.get("accepted") is True, "frozen audit did not pass")
    require(audit.get("checkpoint_sha256") == source_hash, "audit source mismatch")
    require(audit.get("artifact_sha256") == artifact_hash, "audit artifact mismatch")
    require(audit.get("checkpoint_mutated") is False, "audit mutated checkpoint")
    require(
        audit.get("accepted_coverage_changed") is False,
        "audit changed accepted coverage",
    )

    target_name = artifact["target_name"]
    require(target_name in payload["matrices"], "target missing from checkpoint")
    entry = payload["matrices"][target_name]
    shape = tuple(entry["shape"])
    group_size = int(entry["group_size"])
    source_mask = entry["committed_mask"].bool()
    artifact_mask = artifact["committed_mask"].bool()
    artifact_codes = artifact["ternary_codes_int8"].to(torch.int8)
    artifact_scales = artifact["scales_fp16"].half()
    source_codes = grouped(entry["ternary_codes_int8"].to(torch.int8), shape, group_size)
    source_scales = entry["scales_fp16"].half()
    require(torch.equal(source_mask, artifact_mask), "committed mask mismatch")
    require(artifact_codes.shape == source_codes.shape, "code shape mismatch")
    require(artifact_scales.shape == source_scales.shape, "scale shape mismatch")
    require(
        set(artifact_codes.unique().tolist()) <= {-1, 0, 1},
        "artifact contains non-ternary codes",
    )
    require(torch.isfinite(artifact_scales).all(), "artifact scales are not finite")
    require((artifact_scales > 0).all(), "artifact scales are not positive")
    require(
        torch.equal(artifact_codes[~source_mask], source_codes[~source_mask]),
        "artifact changes uncommitted codes",
    )
    require(
        torch.equal(artifact_scales[~source_mask], source_scales[~source_mask]),
        "artifact changes uncommitted scales",
    )
    changed_codes = int((artifact_codes[source_mask] != source_codes[source_mask]).sum())
    changed_scales = int(
        (artifact_scales[source_mask] != source_scales[source_mask]).sum()
    )
    require(changed_codes == 0, "coverage-neutral recode unexpectedly changed codes")
    require(changed_scales > 0, "artifact does not change any committed scale")
    coverage_before = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )

    entry["ternary_codes_int8"] = artifact_codes.reshape(shape[0], -1)[
        :, : shape[1]
    ].contiguous()
    entry["scales_fp16"] = artifact_scales.contiguous()
    coverage_after = sum(
        int(matrix["committed_mask"].sum().item()) * int(matrix["group_size"])
        for matrix in payload["matrices"].values()
    )
    require(coverage_after == coverage_before, "accepted coverage changed")
    previous_metadata = payload.get("metadata", {})
    payload["parent_sha256"] = source_hash
    payload["metadata"] = {
        "experiment": args.tag,
        "mode": "coverage-neutral counterfactual-teacher residual distilled into strict Q2-g128 scales",
        "source_checkpoint_sha256": source_hash,
        "target_name": target_name,
        "ternary_recode_artifact_sha256": artifact_hash,
        "development_result_sha256": development_hash,
        "frozen_audit_result_sha256": audit_hash,
        "commit_policy_sha256": policy_hash,
        "recipe": artifact["recipe"],
        "residual_gain": artifact["residual_gain"],
        "changed_code_values": changed_codes,
        "changed_scale_groups": changed_scales,
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "persistent_bf16_residual": False,
        "previous_metadata": previous_metadata,
    }
    atomic_torch_save(payload, output)
    output_hash = sha256_file(output)
    result = {
        "schema": "wal-tat-ternary-recode-commit-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_hash,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "development_result": str(development_path),
        "development_result_sha256": development_hash,
        "audit_result": str(audit_path),
        "audit_result_sha256": audit_hash,
        "policy": str(policy_path),
        "policy_sha256": policy_hash,
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": output_hash,
        "target_name": target_name,
        "changed_code_values": changed_codes,
        "changed_scale_groups": changed_scales,
        "accepted_weights_before": coverage_before,
        "accepted_weights_after": coverage_after,
        "coverage_changed": False,
        "old_checkpoint_deleted": False,
        "fresh_verification_required": True,
    }
    result_path = PROJECT / f"results/{args.tag}_commit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
