"""Build recovery/dev data from selected domain sources without audit overlap.

The suite can match either validation or train sources while using declared
token ranges. It is development data and must never be described as sealed
audit evidence.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoTokenizer

from build_audit_holdout import (
    arrow_column,
    c4_arrow_pattern,
    code_source_directories,
    one_file,
)
from compensation_window import WORKSPACE, default_model_path, sha256_file


def tokenize(tokenizer, texts) -> torch.Tensor:
    return tokenizer(
        "\n\n".join(text for text in texts if text.strip()), return_tensors="pt"
    ).input_ids[0]


def windows(ids: torch.Tensor, *, count: int, length: int, offset: int):
    if count < 1 or length < 1 or offset < 0:
        raise ValueError("count/length must be positive and offset non-negative")
    needed = offset + count * length + 1
    if ids.numel() < needed:
        raise RuntimeError(f"source has {ids.numel()} tokens, need {needed}")
    return [
        ids[offset + index * length : offset + (index + 1) * length + 1].clone()
        for index in range(count)
    ]


def code_corpus(source: str) -> tuple[list[Path], list[str]]:
    roots = code_source_directories(source)
    paths = []
    for root in roots:
        paths.extend(sorted(root.rglob("*.py")))
    paths = paths[:400]
    texts = [
        f"# source: {path}\n{path.read_text(encoding='utf-8', errors='ignore')}"
        for path in paths
    ]
    if not texts:
        raise RuntimeError(f"no Python files found for {source}")
    return paths, texts


def interleave(calibration_by_domain: dict[str, list[torch.Tensor]], repeats: dict[str, int], base_count: int):
    result = []
    for index in range(base_count):
        for domain in calibration_by_domain:
            for repeat in range(repeats[domain]):
                result.append(calibration_by_domain[domain][repeat * base_count + index])
    return result


def resolved_domain_offsets(args, domains: dict[str, str]) -> tuple[dict, dict]:
    """Resolve optional per-source offsets while preserving the old CLI."""
    calibration = {
        domains[family]: (
            getattr(args, f"{family}_calibration_offset")
            if getattr(args, f"{family}_calibration_offset") is not None
            else args.calibration_offset
        )
        for family in domains
    }
    gates = {
        domains[family]: (
            getattr(args, f"{family}_gate_offset")
            if getattr(args, f"{family}_gate_offset") is not None
            else args.gate_offset
        )
        for family in domains
    }
    return calibration, gates


def resolved_calibration_strides(args, domains: dict[str, str]) -> dict:
    return {
        domains[family]: getattr(args, f"{family}_calibration_stride")
        for family in domains
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-matched-validation-recovery-v1.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument(
        "--c4-split", choices=("validation", "train"), default="validation"
    )
    parser.add_argument(
        "--c4-train-shard",
        type=int,
        default=0,
        help="Zero-based cached C4 train Arrow shard; ignored for validation.",
    )
    parser.add_argument("--c4-text-start", type=int, default=0)
    parser.add_argument("--c4-text-count", type=int, default=4000)
    parser.add_argument(
        "--squad-split", choices=("validation", "train"), default="validation"
    )
    parser.add_argument("--calibration-per-domain", type=int, default=64)
    parser.add_argument("--c4-calibration-repeat", type=int, default=1)
    parser.add_argument("--squad-calibration-repeat", type=int, default=4)
    parser.add_argument("--code-calibration-repeat", type=int, default=1)
    parser.add_argument("--gates-per-domain", type=int, default=256)
    parser.add_argument("--calibration-offset", type=int, default=1_100_003)
    parser.add_argument("--gate-offset", type=int, default=1_200_003)
    for family in ("c4", "squad", "code"):
        parser.add_argument(
            f"--{family}-calibration-offset",
            type=int,
            help="Override --calibration-offset for this source only.",
        )
        parser.add_argument(
            f"--{family}-gate-offset",
            type=int,
            help="Override --gate-offset for this source only.",
        )
        parser.add_argument(
            f"--{family}-calibration-stride",
            type=int,
            help=(
                "Separate repeated calibration segments by this many tokens; "
                "the default keeps them contiguous."
            ),
        )
    parser.add_argument(
        "--code-source",
        choices=("torch", "transformers", "numpy", "datasets"),
        default="torch",
    )
    parser.add_argument("--interleave-calibration", action="store_true")
    parser.add_argument(
        "--policy",
        default=(
            "matched validation-domain recovery/development data; disjoint "
            "declared ranges from audit v3-v6; never final audit"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.c4_text_start < 0 or args.c4_text_count < 1:
        raise ValueError("C4 text start/count must be non-negative/positive")
    model_path = (args.model_path or default_model_path()).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if args.c4_split == "validation":
        c4_path = one_file("allenai___c4/default-c*/0.0.0/*/c4-validation.arrow")
    else:
        c4_path = one_file(c4_arrow_pattern("train", args.c4_train_shard))
    squad_path = one_file(
        f"squad/plain_text/0.0.0/*/squad-{args.squad_split}.arrow"
    )
    all_c4_texts = arrow_column(c4_path, "text")
    c4_texts = all_c4_texts[
        args.c4_text_start : args.c4_text_start + args.c4_text_count
    ]
    squad_texts = arrow_column(squad_path, "context")
    code_paths, code_texts = code_corpus(args.code_source)
    c4_domain = f"c4_{args.c4_split}"
    squad_domain = (
        "squad_context"
        if args.squad_split == "validation"
        else "squad_train_context"
    )
    code_domain = f"{args.code_source}_code"
    domains = {
        "c4": c4_domain,
        "squad": squad_domain,
        "code": code_domain,
    }
    calibration_offsets, gate_offsets = resolved_domain_offsets(args, domains)
    calibration_strides = resolved_calibration_strides(args, domains)
    token_sources = {
        c4_domain: tokenize(tokenizer, c4_texts),
        squad_domain: tokenize(tokenizer, squad_texts),
        code_domain: tokenize(tokenizer, code_texts),
    }
    repeats = {
        c4_domain: args.c4_calibration_repeat,
        squad_domain: args.squad_calibration_repeat,
        code_domain: args.code_calibration_repeat,
    }
    calibration_by_domain = {}
    calibration_ranges = {}
    gates = {}
    for domain, ids in token_sources.items():
        segment_span = args.calibration_per_domain * args.length
        requested_stride = calibration_strides[domain]
        stride = requested_stride if requested_stride is not None else segment_span
        if stride < segment_span:
            raise ValueError(f"calibration segments overlap for {domain}")
        segments = [
            [
                calibration_offsets[domain] + repeat * stride,
                calibration_offsets[domain] + repeat * stride + segment_span,
            ]
            for repeat in range(repeats[domain])
        ]
        gate_end = gate_offsets[domain] + args.gates_per_domain * args.length
        for start, end in segments:
            if max(start, gate_offsets[domain]) < min(end, gate_end):
                raise ValueError(f"calibration and gate ranges overlap for {domain}")
        calibration_by_domain[domain] = []
        for start, _ in segments:
            calibration_by_domain[domain].extend(
                windows(
                    ids,
                    count=args.calibration_per_domain,
                    length=args.length,
                    offset=start,
                )
            )
        calibration_ranges[domain] = (
            segments
            if requested_stride is not None
            else [segments[0][0], segments[-1][1]]
        )
        gates[domain] = windows(
            ids,
            count=args.gates_per_domain,
            length=args.length,
            offset=gate_offsets[domain],
        )
    if args.interleave_calibration:
        calibration = interleave(
            calibration_by_domain, repeats, args.calibration_per_domain
        )
    else:
        calibration = [
            value
            for domain in calibration_by_domain
            for value in calibration_by_domain[domain]
        ]

    sources = [c4_path, squad_path, *code_paths]
    ranges = {
        "calibration": {
            domain: calibration_ranges[domain]
            for domain in token_sources
        },
        "gates": {
            domain: [
                gate_offsets[domain],
                gate_offsets[domain] + args.gates_per_domain * args.length,
            ]
            for domain in token_sources
        },
    }
    payload = {
        "format": "wal-tat-matched-validation-recovery-v1",
        "policy": args.policy,
        "model_revision": model_path.name,
        "sequence_length": args.length,
        "calibration_per_domain": args.calibration_per_domain,
        "c4_calibration_repeat": args.c4_calibration_repeat,
        "squad_calibration_repeat": args.squad_calibration_repeat,
        "code_calibration_repeat": args.code_calibration_repeat,
        "gates_per_domain": args.gates_per_domain,
        "calibration_offset": args.calibration_offset,
        "gate_offset": args.gate_offset,
        "domain_calibration_offsets": calibration_offsets,
        "domain_gate_offsets": gate_offsets,
        "domain_calibration_strides": calibration_strides,
        "interleave_calibration": args.interleave_calibration,
        "source_splits": {
            c4_domain: f"c4 {args.c4_split}",
            squad_domain: f"squad {args.squad_split}",
            code_domain: f"installed {args.code_source} Python source",
        },
        "data_splits": {
            "c4": args.c4_split,
            "squad": args.squad_split,
        },
        "c4_train_shard": (
            args.c4_train_shard if args.c4_split == "train" else None
        ),
        "c4_text_start": args.c4_text_start,
        "c4_text_count": args.c4_text_count,
        "code_source": args.code_source,
        "ranges": ranges,
        "calibration": calibration,
        "gates": gates,
        "sources": [str(path) for path in sources],
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "full_token_stream_sha256": {
            domain: hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
            for domain, ids in token_sources.items()
        },
        "token_sha256": {
            domain: hashlib.sha256(torch.stack(values).numpy().tobytes()).hexdigest()
            for domain, values in gates.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "calibration_sequences": len(calibration),
            "gate_sequences": {domain: len(values) for domain, values in gates.items()},
            "ranges": ranges,
        }
    )


if __name__ == "__main__":
    main()
