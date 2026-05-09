#!/usr/bin/env python3
"""WAL CLI — Unified command-line interface for all 11 phases."""
import argparse
import sys
from pathlib import Path


def cmd_encode(args):
    """Phase 1: Encode a model to WAL format."""
    from .encode import encode_model
    print(f"[WAL] Encoding {args.input} → {args.output}")
    encode_model(args.input, args.output, K=args.K, C=args.C, device=args.device, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done. Output: {args.output}")


def cmd_decode(args):
    """Phase 3: Decode WAL model to dense weights."""
    from .decode import decode_model
    print(f"[WAL] Decoding {args.input} → {args.output}")
    decode_model(args.input, args.output, device=args.device, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done. Output: {args.output}")


def cmd_grammar(args):
    """Phase 2: Convert between text and binary formats."""
    from .grammar import convert_format
    print(f"[WAL] Converting {args.input} → {args.output} ({args.to})")
    convert_format(args.input, args.output, target_format=args.to)
    print(f"[WAL] Done.")


def cmd_compress(args):
    """Phase 4: Binary compression."""
    from .compress import compress_wal
    print(f"[WAL] Compressing {args.input} → {args.output}")
    compress_wal(args.input, args.output, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done.")


def cmd_hierarchy(args):
    """Phase 5: Build hierarchical atoms."""
    from .hierarchy import build_hierarchy
    print(f"[WAL] Building L1 atoms for {args.input}")
    build_hierarchy(args.input, args.output, max_l1=args.max_l1, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done.")


def cmd_torch(args):
    """Phase 6: PyTorch integration."""
    from .torch import replace_linear
    print(f"[WAL] Replacing nn.Linear with WAL in {args.input}")
    replace_linear(args.input, args.output, K=args.K, C=args.C, cached=args.cached, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done.")


def cmd_debug(args):
    """Phase 7: Interactive debugger."""
    from .debug import debug_wal
    debug_wal(args.input, index=args.index, trust_pickle=args.trust_pickle)


def cmd_library(args):
    """Phase 8: Atom library operations."""
    from .library import library_op
    library_op(args.command, args.input, args.output, name=args.name)


def cmd_backend(args):
    """Phase 9: Backend selection and benchmark."""
    from .backend import benchmark_backend
    benchmark_backend(backend=args.backend, device=args.device)


def cmd_meta(args):
    """Phase 10: Meta-learning operations."""
    from .meta import meta_op
    meta_op(args.command, args.input, args.output, rank=args.rank, alpha=args.alpha)


def cmd_export(args):
    """Phase 11: Export to ONNX or HF Hub."""
    from .export import export_model
    print(f"[WAL] Exporting {args.input} → {args.output} ({args.format})")
    export_model(args.input, args.output, fmt=args.format, dummy_shape=args.dummy_shape, trust_pickle=args.trust_pickle)
    print(f"[WAL] Done.")


def cmd_merge(args):
    """Phase 11: Merge WAL models."""
    from .merge import merge_models
    print(f"[WAL] Merging {len(args.inputs)} models → {args.output}")
    merge_models(args.inputs, args.output, method=args.method, weights=args.weights)
    print(f"[WAL] Done.")


def cmd_pipeline(args):
    """Full pipeline: encode → optimize → export."""
    from .pipeline import run_pipeline
    print(f"[WAL] Running full pipeline on {args.input}")
    run_pipeline(
        args.input, args.output,
        K=args.K, C=args.C,
        export_format=args.export_format,
        device=args.device,
        trust_pickle=args.trust_pickle,
    )
    print(f"[WAL] Pipeline complete. Output: {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wal",
        description="Weight Assembly Language — a language for neural network weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wal encode model.pt --output wal_model/
  wal decode wal_model/ --output dense.pt
  wal export wal_model/ --format onnx --output model.onnx
  wal merge model_a/ model_b/ --method soup --output merged/
  wal pipeline model.pt --output shipped/
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # encode
    p = subparsers.add_parser("encode", help="Encode a model to WAL format")
    p.add_argument("input", help="Input model path (.pt or .safetensors)")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--K", type=int, default=256, help="Number of atoms")
    p.add_argument("--C", type=int, default=16, help="Number of coefficients")
    p.add_argument("--device", default="cuda", help="Device for encoding")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_encode)

    # decode
    p = subparsers.add_parser("decode", help="Decode WAL model to dense weights")
    p.add_argument("input", help="Input WAL directory")
    p.add_argument("--output", "-o", required=True, help="Output model path")
    p.add_argument("--device", default="cuda", help="Device for decoding")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_decode)

    # grammar
    p = subparsers.add_parser("grammar", help="Convert between text and binary formats")
    p.add_argument("input", help="Input file")
    p.add_argument("--output", "-o", required=True, help="Output file")
    p.add_argument("--to", choices=["text", "binary"], required=True, help="Target format")
    p.set_defaults(func=cmd_grammar)

    # compress
    p = subparsers.add_parser("compress", help="Compress WAL to binary")
    p.add_argument("input", help="Input WAL directory")
    p.add_argument("--output", "-o", required=True, help="Output binary file")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_compress)

    # hierarchy
    p = subparsers.add_parser("hierarchy", help="Build hierarchical atoms")
    p.add_argument("input", help="Input WAL directory")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--max-l1", type=int, default=64, help="Max L1 atoms")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_hierarchy)

    # torch
    p = subparsers.add_parser("torch", help="Replace nn.Linear with WAL layers")
    p.add_argument("input", help="Input model path")
    p.add_argument("--output", "-o", required=True, help="Output model path")
    p.add_argument("--K", type=int, default=256, help="Number of atoms")
    p.add_argument("--C", type=int, default=16, help="Number of coefficients")
    p.add_argument("--cached", action="store_true", help="Use WALCachedLinear")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_torch)

    # debug
    p = subparsers.add_parser("debug", help="Interactive debugger")
    p.add_argument("input", help="Input WAL file")
    p.add_argument("--index", type=int, default=0, help="Weight index to debug")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_debug)

    # library
    p = subparsers.add_parser("library", help="Atom library operations")
    p.add_argument("command", choices=["create", "add", "query", "save", "load"], help="Operation")
    p.add_argument("input", nargs="?", help="Input file/directory")
    p.add_argument("--output", "-o", help="Output directory")
    p.add_argument("--name", help="Entry name")
    p.set_defaults(func=cmd_library)

    # backend
    p = subparsers.add_parser("backend", help="Backend selection and benchmark")
    p.add_argument("--backend", default="auto", help="Backend name (auto/cpu/cuda/...)")
    p.add_argument("--device", default="cuda", help="Device")
    p.set_defaults(func=cmd_backend)

    # meta
    p = subparsers.add_parser("meta", help="Meta-learning operations")
    p.add_argument("command", choices=["adapter", "soup", "evolve"], help="Operation")
    p.add_argument("input", nargs="+", help="Input model(s)")
    p.add_argument("--output", "-o", help="Output model")
    p.add_argument("--rank", type=int, default=4, help="Adapter rank")
    p.add_argument("--alpha", type=float, default=1.0, help="Adapter alpha")
    p.set_defaults(func=cmd_meta)

    # export
    p = subparsers.add_parser("export", help="Export to ONNX or HF Hub")
    p.add_argument("input", help="Input WAL directory")
    p.add_argument("--output", "-o", required=True, help="Output file")
    p.add_argument("--format", choices=["onnx", "hub"], required=True, help="Export format")
    p.add_argument("--dummy-shape", default="1,4096", help="Dummy input shape for ONNX")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_export)

    # merge
    p = subparsers.add_parser("merge", help="Merge WAL models")
    p.add_argument("inputs", nargs="+", help="Input models")
    p.add_argument("--output", "-o", required=True, help="Output model")
    p.add_argument("--method", choices=["soup", "linear", "slerp", "ties"], default="soup", help="Merge method")
    p.add_argument("--weights", type=float, nargs="+", help="Per-model weights")
    p.set_defaults(func=cmd_merge)

    # pipeline
    p = subparsers.add_parser("pipeline", help="Full pipeline: encode → optimize → export")
    p.add_argument("input", help="Input model path")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--K", type=int, default=256, help="Number of atoms")
    p.add_argument("--C", type=int, default=16, help="Number of coefficients")
    p.add_argument("--export-format", choices=["onnx", "hub", "none"], default="none", help="Export format")
    p.add_argument("--device", default="cuda", help="Device")
    p.add_argument("--trust-pickle", action="store_true", help="Allow pickle-backed torch.load for trusted files")
    p.set_defaults(func=cmd_pipeline)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
