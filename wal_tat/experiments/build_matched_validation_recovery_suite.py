"""Build recovery/dev data from validation-domain sources without audit overlap.

The suite intentionally matches the source *types* used by rotating audit v5
(C4 validation, SQuAD validation contexts and PyTorch source) while using
different declared token ranges.  It is development data and must never be
described as sealed audit evidence.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from build_audit_holdout import arrow_column, one_file
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
    if source == "torch":
        root = Path(torch.__file__).resolve().parent
    elif source == "transformers":
        root = Path(transformers.__file__).resolve().parent
    else:
        raise ValueError(f"unsupported code source: {source}")
    paths = sorted(root.rglob("*.py"))[:400]
    texts = [
        f"# source: {path}\n{path.read_text(encoding='utf-8', errors='ignore')}"
        for path in paths
    ]
    if not texts:
        raise RuntimeError(f"no Python files found under {root}")
    return paths, texts


def interleave(calibration_by_domain: dict[str, list[torch.Tensor]], repeats: dict[str, int], base_count: int):
    result = []
    for index in range(base_count):
        for domain in calibration_by_domain:
            for repeat in range(repeats[domain]):
                result.append(calibration_by_domain[domain][repeat * base_count + index])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-matched-validation-recovery-v1.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--calibration-per-domain", type=int, default=64)
    parser.add_argument("--c4-calibration-repeat", type=int, default=1)
    parser.add_argument("--squad-calibration-repeat", type=int, default=4)
    parser.add_argument("--code-calibration-repeat", type=int, default=1)
    parser.add_argument("--gates-per-domain", type=int, default=256)
    parser.add_argument("--calibration-offset", type=int, default=1_100_003)
    parser.add_argument("--gate-offset", type=int, default=1_200_003)
    parser.add_argument(
        "--code-source", choices=("torch", "transformers"), default="torch"
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
    if args.calibration_offset == args.gate_offset:
        raise ValueError("calibration and gate offsets must differ")
    model_path = (args.model_path or default_model_path()).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    c4_path = one_file("allenai___c4/default-c*/0.0.0/*/c4-validation.arrow")
    squad_path = one_file("squad/plain_text/0.0.0/*/squad-validation.arrow")
    c4_texts = arrow_column(c4_path, "text")[:4000]
    squad_texts = arrow_column(squad_path, "context")
    code_paths, code_texts = code_corpus(args.code_source)
    code_domain = f"{args.code_source}_code"
    token_sources = {
        "c4_validation": tokenize(tokenizer, c4_texts),
        "squad_context": tokenize(tokenizer, squad_texts),
        code_domain: tokenize(tokenizer, code_texts),
    }
    repeats = {
        "c4_validation": args.c4_calibration_repeat,
        "squad_context": args.squad_calibration_repeat,
        code_domain: args.code_calibration_repeat,
    }
    calibration_by_domain = {}
    gates = {}
    for domain, ids in token_sources.items():
        calibration_by_domain[domain] = windows(
            ids,
            count=args.calibration_per_domain * repeats[domain],
            length=args.length,
            offset=args.calibration_offset,
        )
        gates[domain] = windows(
            ids,
            count=args.gates_per_domain,
            length=args.length,
            offset=args.gate_offset,
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
            domain: [
                args.calibration_offset,
                args.calibration_offset
                + args.calibration_per_domain * repeats[domain] * args.length,
            ]
            for domain in token_sources
        },
        "gates": {
            domain: [
                args.gate_offset,
                args.gate_offset + args.gates_per_domain * args.length,
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
        "interleave_calibration": args.interleave_calibration,
        "source_splits": {
            "c4_validation": "c4 validation",
            "squad_context": "squad validation",
            code_domain: f"installed {args.code_source} Python source",
        },
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
