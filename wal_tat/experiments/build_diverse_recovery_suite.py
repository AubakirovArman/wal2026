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
    texts = [
        f"# source: {path.relative_to(WORKSPACE)}\n{path.read_text(encoding='utf-8', errors='ignore')}"
        for path in paths
    ]
    return paths, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-diverse-recovery-v1-l256.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--calibration-per-domain", type=int, default=64)
    parser.add_argument("--c4-calibration-repeat", type=int, default=1)
    parser.add_argument("--squad-calibration-repeat", type=int, default=1)
    parser.add_argument("--code-calibration-repeat", type=int, default=1)
    parser.add_argument("--gates-per-domain", type=int, default=256)
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
    code_paths, code_values = code_texts(
        (
            WORKSPACE / "wal2/vendor/Hestia",
            WORKSPACE / "wal2/vendor/TWLA",
            WORKSPACE / "wal2/vendor/GSQ",
        )
    )

    token_sources = {
        "c4_train": tokenize(tokenizer, c4_texts),
        "squad_train": tokenize(tokenizer, squad_texts),
        "vendor_code_train": tokenize(tokenizer, code_values),
    }
    calibration = []
    gates = {}
    for domain, ids in token_sources.items():
        repeats = {
            "c4_train": args.c4_calibration_repeat,
            "squad_train": args.squad_calibration_repeat,
            "vendor_code_train": args.code_calibration_repeat,
        }
        calibration_count = args.calibration_per_domain * repeats[domain]
        calibration.extend(
            windows(
                ids,
                count=calibration_count,
                length=args.length,
                offset=10007,
            )
        )
        gates[domain] = windows(
            ids,
            count=args.gates_per_domain,
            length=args.length,
            offset=80003,
        )

    source_paths = [*c4_paths, squad_path, *code_paths]
    payload = {
        "format": "wal-tat-diverse-recovery-v1",
        "policy": "training calibration and development gates; never final audit",
        "model_revision": model_path.name,
        "sequence_length": args.length,
        "calibration_per_domain": args.calibration_per_domain,
        "c4_calibration_repeat": args.c4_calibration_repeat,
        "squad_calibration_repeat": args.squad_calibration_repeat,
        "code_calibration_repeat": args.code_calibration_repeat,
        "gates_per_domain": args.gates_per_domain,
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
