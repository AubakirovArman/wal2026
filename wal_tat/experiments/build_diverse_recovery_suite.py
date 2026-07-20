"""Build a diverse recovery/dev suite disjoint from audit validation splits."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import torch
from transformers import AutoTokenizer

from compensation_window import WORKSPACE, default_model_path, sha256_file


DATASET_CACHE = Path("/home/arman/.cache/huggingface/datasets")


def arrow_column(path: Path, name: str):
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    return table[name].to_pylist()


def one_file(pattern: str) -> Path:
    matches = sorted(DATASET_CACHE.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source for {pattern!r}, found {matches}")
    return matches[0]


def tokenize(tokenizer, texts):
    return tokenizer(
        "\n\n".join(text for text in texts if text.strip()), return_tensors="pt"
    ).input_ids[0]


def windows(ids: torch.Tensor, *, count: int, length: int, offset: int):
    needed = offset + count * length + 1
    if ids.numel() < needed:
        raise RuntimeError(f"source has {ids.numel()} tokens, need {needed}")
    return [
        ids[offset + index * length : offset + (index + 1) * length + 1].clone()
        for index in range(count)
    ]


def code_texts(directories):
    paths = []
    for directory in directories:
        paths.extend(sorted(directory.rglob("*.py")))
    texts = []
    for path in paths:
        try:
            source_name = path.relative_to(WORKSPACE)
        except ValueError:
            source_name = path
        texts.append(
            f"# source: {source_name}\n"
            f"{path.read_text(encoding='utf-8', errors='ignore')}"
        )
    return paths, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-diverse-recovery-v2-l256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--calibration-per-domain", type=int, default=64)
    parser.add_argument("--c4-calibration-repeat", type=int, default=1)
    parser.add_argument("--squad-calibration-repeat", type=int, default=1)
    parser.add_argument("--code-calibration-repeat", type=int, default=1)
    parser.add_argument("--gates-per-domain", type=int, default=256)
    parser.add_argument("--calibration-offset", type=int, default=10007)
    parser.add_argument("--gate-offset", type=int, default=80003)
    parser.add_argument(
        "--code-directory",
        type=Path,
        action="append",
        help="Python source directory; repeat to concatenate several disjoint corpora",
    )
    parser.add_argument("--code-domain", default="vendor_code_train")
    parser.add_argument(
        "--interleave-calibration",
        action="store_true",
        help="interleave weighted domain windows instead of concatenating domains",
    )
    parser.add_argument(
        "--policy",
        default="training calibration and development gates; never final audit",
    )
    args = parser.parse_args()
    model_path = (args.model_path or default_model_path()).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    c4_paths = sorted(
        DATASET_CACHE.glob("allenai___c4/default-b*/0.0.0/*/c4-train-*.arrow")
    )
    if len(c4_paths) != 2:
        raise RuntimeError(f"expected two cached C4 train shards, found {c4_paths}")
    c4_texts = []
    for path in c4_paths:
        c4_texts.extend(arrow_column(path, "text")[:2500])
    squad_path = one_file("squad/plain_text/0.0.0/*/squad-train.arrow")
    squad_texts = arrow_column(squad_path, "context")
    code_directories = args.code_directory or [
        WORKSPACE / "wal2/vendor/Hestia",
        WORKSPACE / "wal2/vendor/TWLA",
        WORKSPACE / "wal2/vendor/GSQ",
    ]
    code_directories = [path.resolve() for path in code_directories]
    if any(not path.is_dir() for path in code_directories):
        raise RuntimeError(f"invalid code directories: {code_directories}")
    code_paths, code_values = code_texts(code_directories)

    token_sources = {
        "c4_train": tokenize(tokenizer, c4_texts),
        "squad_train": tokenize(tokenizer, squad_texts),
        args.code_domain: tokenize(tokenizer, code_values),
    }
    calibration_by_domain = {}
    gates = {}
    repeats = {
        "c4_train": args.c4_calibration_repeat,
        "squad_train": args.squad_calibration_repeat,
        args.code_domain: args.code_calibration_repeat,
    }
    for domain, ids in token_sources.items():
        calibration_count = args.calibration_per_domain * repeats[domain]
        calibration_by_domain[domain] = windows(
            ids,
            count=calibration_count,
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
        calibration = []
        for index in range(args.calibration_per_domain):
            for domain in token_sources:
                for repeat in range(repeats[domain]):
                    source_index = repeat * args.calibration_per_domain + index
                    calibration.append(calibration_by_domain[domain][source_index])
    else:
        calibration = [
            window
            for domain in token_sources
            for window in calibration_by_domain[domain]
        ]

    source_paths = [*c4_paths, squad_path, *code_paths]
    payload = {
        "format": "wal-tat-diverse-recovery-v2",
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
        "code_directories": [str(path) for path in code_directories],
        "code_domain": args.code_domain,
        "ranges": {
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
        },
        "calibration": calibration,
        "gates": gates,
        "sources": [str(path) for path in source_paths],
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "token_sha256": {
            name: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            for name, ids in token_sources.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "calibration_tokens": len(calibration) * args.length,
            "gate_tokens": {
                name: len(chunks) * args.length for name, chunks in gates.items()
            },
        }
    )


if __name__ == "__main__":
    main()
