"""Build a deterministic audit-only suite disjoint from WAL-TAT development gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import datasets
import numpy
import pyarrow as pa
import torch
import transformers
from transformers import AutoTokenizer

from compensation_window import WORKSPACE, default_model_path, sha256_file


DATASET_CACHE = Path("/home/arman/.cache/huggingface/datasets")
OPTIONAL_CODE_SOURCES = ("scipy", "pandas", "sklearn")


def one_file(pattern: str) -> Path:
    matches = sorted(DATASET_CACHE.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one cached source for {pattern!r}, found {matches}")
    return matches[0]


def c4_arrow_pattern(split: str, train_shard: int = 0) -> str:
    if split == "validation":
        return "allenai___c4/default-c*/0.0.0/*/c4-validation.arrow"
    if split != "train":
        raise ValueError(f"unsupported C4 split: {split}")
    if train_shard < 0:
        raise ValueError("C4 train shard must be non-negative")
    return (
        "allenai___c4/default-b*/0.0.0/*/"
        f"c4-train-{train_shard:05d}-of-*.arrow"
    )


def token_stream(tokenizer, texts):
    return tokenizer(
        "\n\n".join(text for text in texts if text.strip()),
        return_tensors="pt",
    ).input_ids[0]


def text_windows(ids, *, count: int, length: int, offset: int):
    needed = offset + count * length + 1
    if ids.numel() < needed:
        raise RuntimeError(f"source has {ids.numel()} tokens, but {needed} are required")
    return [
        ids[offset + index * length : offset + (index + 1) * length + 1].clone()
        for index in range(count)
    ]


def arrow_column(path: Path, name: str):
    """Read a Hugging Face Arrow stream without decoding versioned metadata."""
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    return table[name].to_pylist()


def code_source_directories(source: str) -> tuple[Path, ...]:
    if source == "vendor":
        return (
            WORKSPACE / "wal2/vendor/Hestia",
            WORKSPACE / "wal2/vendor/TWLA",
            WORKSPACE / "wal2/vendor/GSQ",
        )
    roots = {
        "transformers": Path(transformers.__file__).resolve().parent,
        "torch": Path(torch.__file__).resolve().parent,
        "numpy": Path(numpy.__file__).resolve().parent,
        "datasets": Path(datasets.__file__).resolve().parent,
    }
    if source in OPTIONAL_CODE_SOURCES:
        spec = importlib.util.find_spec(source)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"optional code source {source!r} is not installed")
        return (Path(spec.origin).resolve().parent,)
    if source not in roots:
        raise ValueError(f"unsupported code source: {source}")
    return (roots[source],)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "wal2/cache/wal-tat-audit-holdout-v1-l256-g512.pt",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument(
        "--c4-split",
        choices=("validation", "train"),
        default="validation",
        help="Use a fresh raw Arrow source after the validation stream is exhausted.",
    )
    parser.add_argument(
        "--squad-split",
        choices=("validation", "train"),
        default="validation",
        help="Use a fresh raw Arrow source after the validation stream is exhausted.",
    )
    parser.add_argument(
        "--c4-train-shard",
        type=int,
        default=0,
        help="Zero-based cached C4 train Arrow shard; ignored for validation.",
    )
    parser.add_argument("--c4-text-start", type=int, default=0)
    parser.add_argument("--c4-text-count", type=int, default=4000)
    parser.add_argument("--c4-offset", type=int, default=17011)
    parser.add_argument("--squad-offset", type=int, default=9011)
    parser.add_argument("--code-offset", type=int, default=4001)
    parser.add_argument(
        "--code-source",
        choices=(
            "vendor",
            "transformers",
            "torch",
            "numpy",
            "datasets",
            *OPTIONAL_CODE_SOURCES,
        ),
        default="vendor",
    )
    parser.add_argument(
        "--policy",
        default="audit-only; never use for transaction selection or rollback",
        help="Immutable role recorded in the suite; declare recurring validation explicitly.",
    )
    args = parser.parse_args()
    if args.c4_text_start < 0 or args.c4_text_count < 1:
        raise ValueError("C4 text start/count must be non-negative/positive")
    model_path = (args.model_path or default_model_path()).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    c4_path = one_file(c4_arrow_pattern(args.c4_split, args.c4_train_shard))
    squad_path = one_file(
        f"squad/plain_text/0.0.0/*/squad-{args.squad_split}.arrow"
    )
    c4_texts = arrow_column(c4_path, "text")
    squad_contexts = arrow_column(squad_path, "context")

    code_files = []
    directories = code_source_directories(args.code_source)
    for directory in directories:
        code_files.extend(sorted(directory.rglob("*.py")))
    if args.code_source != "vendor":
        code_files = code_files[:400]
    if not code_files:
        raise RuntimeError("no vendor Python sources found")
    code_texts = []
    for path in code_files:
        label = path.relative_to(WORKSPACE) if path.is_relative_to(WORKSPACE) else path
        code_texts.append(
            f"# source: {label}\n{path.read_text(encoding='utf-8', errors='ignore')}"
        )

    c4_domain = f"c4_{args.c4_split}"
    squad_domain = (
        "squad_context"
        if args.squad_split == "validation"
        else "squad_train_context"
    )
    code_domain = f"{args.code_source}_code"
    token_sources = {
        c4_domain: token_stream(
            tokenizer,
            c4_texts[
                args.c4_text_start : args.c4_text_start + args.c4_text_count
            ],
        ),
        squad_domain: token_stream(tokenizer, squad_contexts),
        code_domain: token_stream(tokenizer, code_texts),
    }
    offsets = {
        c4_domain: args.c4_offset,
        squad_domain: args.squad_offset,
        code_domain: args.code_offset,
    }
    gates = {
        c4_domain: text_windows(
            token_sources[c4_domain],
            count=args.sequences,
            length=args.length,
            offset=args.c4_offset,
        ),
        squad_domain: text_windows(
            token_sources[squad_domain],
            count=args.sequences,
            length=args.length,
            offset=args.squad_offset,
        ),
        code_domain: text_windows(
            token_sources[code_domain],
            count=args.sequences,
            length=args.length,
            offset=args.code_offset,
        ),
    }
    source_paths = [c4_path, squad_path, *code_files]
    payload = {
        "format": "wal-tat-audit-holdout-v1",
        "policy": args.policy,
        "model_revision": model_path.name,
        "sequence_length": args.length,
        "sequences_per_domain": args.sequences,
        "predicted_tokens_per_domain": args.length * args.sequences,
        "offsets": {
            c4_domain: args.c4_offset,
            squad_domain: args.squad_offset,
            "code": args.code_offset,
        },
        "ranges": {
            "gates": {
                domain: [offsets[domain], offsets[domain] + args.sequences * args.length]
                for domain in gates
            }
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
        "gates": gates,
        "sources": [str(path) for path in source_paths],
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "full_token_stream_sha256": {
            domain: hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
            for domain, ids in token_sources.items()
        },
        "token_sha256": {
            name: hashlib.sha256(torch.stack(chunks).numpy().tobytes()).hexdigest()
            for name, chunks in gates.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "domains": {name: len(chunks) * args.length for name, chunks in gates.items()},
        }
    )


if __name__ == "__main__":
    main()
