"""Pack or verify one checkpoint matrix with the reference Q2-g binary format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compensation_window import PROJECT, sha256_file
from wal_tat import (
    pack_partial_matrix,
    read_packed_matrix,
    unpack_partial_matrix,
    write_packed_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("matrix")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.matrix not in payload["matrices"]:
        raise ValueError(f"matrix is absent from checkpoint: {args.matrix}")
    entry = payload["matrices"][args.matrix]
    expected = pack_partial_matrix(
        entry["fp_master_bf16"],
        entry["ternary_codes_int8"],
        entry["scales_fp16"],
        entry["committed_mask"],
        group_size=int(entry["group_size"]),
    )
    if output.exists():
        if not args.verify_existing:
            raise FileExistsError(output)
        packed = read_packed_matrix(output)
    else:
        write_packed_matrix(output, expected)
        packed = read_packed_matrix(output)

    packed_fields_exact = (
        packed.shape == expected.shape
        and packed.group_size == expected.group_size
        and packed.padding == expected.padding
        and packed.total_groups == expected.total_groups
        and packed.committed_groups == expected.committed_groups
        and torch.equal(packed.mask_packed, expected.mask_packed)
        and torch.equal(packed.codes_packed, expected.codes_packed)
        and torch.equal(packed.scales_fp16, expected.scales_fp16)
        and torch.equal(
            packed.fallback_bf16.reshape(-1), expected.fallback_bf16.reshape(-1)
        )
    )
    reconstructed = unpack_partial_matrix(packed)
    expected_weight = unpack_partial_matrix(expected)
    roundtrip_exact = torch.equal(reconstructed, expected_weight)
    mask_exact = torch.equal(
        packed.committed_mask().view_as(entry["committed_mask"]),
        entry["committed_mask"],
    )
    if not (packed_fields_exact and roundtrip_exact and mask_exact):
        raise RuntimeError("packed matrix failed exact verification")
    result = {
        "schema": "wal-tat-reference-packed-matrix-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "matrix": args.matrix,
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "artifact_bytes": output.stat().st_size,
        "shape": list(packed.shape),
        "weights": packed.shape[0] * packed.shape[1],
        "group_size": packed.group_size,
        "total_groups": packed.total_groups,
        "committed_groups": packed.committed_groups,
        "coverage": packed.committed_groups / packed.total_groups,
        "payload_bytes": packed.payload_nbytes,
        "header_bytes": packed.serialized_nbytes - packed.payload_nbytes,
        "true_bpw_payload": packed.true_bpw(include_header=False),
        "true_bpw_file": packed.true_bpw(include_header=True),
        "packed_fields_exact": packed_fields_exact,
        "roundtrip_exact": roundtrip_exact,
        "committed_mask_exact": mask_exact,
        "checkpoint_mutated": sha256_file(checkpoint) != checkpoint_hash,
    }
    result_path = PROJECT / f"results/{args.tag}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
