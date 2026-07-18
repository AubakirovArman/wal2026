"""Fresh-process WAL-TAT checkpoint verification on a frozen suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compensation_window import PROJECT, default_model_path, fresh_verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gate-ratio", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model_path = (args.model_path or default_model_path()).resolve()
    suite = torch.load(args.suite, map_location="cpu", weights_only=False)
    result = fresh_verify(args.checkpoint.resolve(), model_path, suite, args)
    result.update(
        {
            "schema": "wal-tat-fresh-verify-v1",
            "suite": str(args.suite.resolve()),
            "suite_sequences": {
                name: len(chunks) for name, chunks in suite["gates"].items()
            },
            "model": str(model_path),
            "gate_ratio": args.gate_ratio,
        }
    )
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
