"""Public WAL command-line entry points."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from framework.cli import main as _framework_main


CORE_COMMANDS = {
    "encode",
    "decode",
    "grammar",
    "compress",
    "hierarchy",
    "torch",
    "debug",
    "library",
    "backend",
    "meta",
    "export",
    "merge",
    "pipeline",
    "validate-results",
}

STUDIO_COMMANDS = {
    "init",
    "edit",
    "status",
    "tag",
    "rollback",
    "build",
    "test",
    "diff",
    "blame",
    "bisect",
}


def _argv(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def core_main(argv: Sequence[str] | None = None) -> None:
    """Run WAL core encode/decode/export/result-validation commands."""
    args = _argv(argv)
    _framework_main(args if args else ["--help"])


def _cmd_studio_init(args: argparse.Namespace) -> None:
    from wal_build import init_project

    init_project(args.base_model)


def _cmd_studio_edit_add(args: argparse.Namespace) -> None:
    from wal_build import add_edit

    add_edit(args.recipe_file, strategy=args.strategy)


def _cmd_studio_status(args: argparse.Namespace) -> None:
    from wal_build import status

    status()


def _cmd_studio_tag(args: argparse.Namespace) -> None:
    from wal_build import tag_version

    tag_version(args.name, build_id=args.build_id)


def _cmd_studio_rollback(args: argparse.Namespace) -> None:
    from wal_build import rollback

    rollback(args.tag)


def _cmd_studio_planned(args: argparse.Namespace) -> None:
    command = getattr(args, "studio_command", "unknown")
    print(
        f"[WAL Studio] `{command}` is a pre-alpha planned CLI command. "
        "Use `python wal_studio_v01/demo.py` for the current end-to-end demo."
    )
    raise SystemExit(2)


def build_studio_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wal studio",
        description="WAL Studio pre-alpha WeightOps commands",
    )
    subparsers = parser.add_subparsers(dest="studio_command")

    init_parser = subparsers.add_parser("init", help="Initialize a WAL Studio project")
    init_parser.add_argument("base_model", help="Base model name or local path")
    init_parser.set_defaults(func=_cmd_studio_init)

    edit_parser = subparsers.add_parser("edit", help="Edit recipe commands")
    edit_subparsers = edit_parser.add_subparsers(dest="edit_command")
    edit_add = edit_subparsers.add_parser("add", help="Add a JSON recipe file")
    edit_add.add_argument("recipe_file", help="Path to recipe JSON")
    edit_add.add_argument("--strategy", default="auto", help="Recipe strategy label")
    edit_add.set_defaults(func=_cmd_studio_edit_add)

    status_parser = subparsers.add_parser("status", help="Show WAL Studio project status")
    status_parser.set_defaults(func=_cmd_studio_status)

    tag_parser = subparsers.add_parser("tag", help="Tag the latest or specified build")
    tag_parser.add_argument("name", help="Tag name")
    tag_parser.add_argument("build_id", nargs="?", help="Optional build id")
    tag_parser.set_defaults(func=_cmd_studio_tag)

    rollback_parser = subparsers.add_parser("rollback", help="Resolve a tag for rollback")
    rollback_parser.add_argument("tag", help="Tag name")
    rollback_parser.set_defaults(func=_cmd_studio_rollback)

    for command in ("build", "test", "diff", "blame", "bisect"):
        planned = subparsers.add_parser(command, help=f"Planned pre-alpha `{command}` command")
        planned.set_defaults(func=_cmd_studio_planned)

    return parser


def studio_main(argv: Sequence[str] | None = None) -> None:
    """Run WAL Studio recipe/build/tag/rollback commands."""
    args = _argv(argv)
    parser = build_studio_parser()
    if not args:
        parser.print_help()
        return
    parsed = parser.parse_args(args)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified WAL CLI.

    New users should prefer explicit namespaces:

    - ``wal core ...`` for WAL encode/decode/export/result validation.
    - ``wal studio ...`` for WeightOps recipe/tag/rollback commands.

    Legacy top-level commands are still accepted for compatibility.
    """
    args = _argv(argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: wal {core,studio,<legacy-command>} ...\n\n"
            "Namespaces:\n"
            "  core      WAL core encode/decode/export/result-validation commands\n"
            "  studio    pre-alpha WeightOps recipe/tag/rollback commands\n\n"
            "Compatibility:\n"
            "  wal validate-results experiments --fail-on-invalid\n"
            "  wal encode model.pt --output wal_model/\n"
            "  wal init base-model\n\n"
            "Run `wal core --help` or `wal studio --help` for details."
        )
        return

    namespace, rest = args[0], args[1:]
    if namespace == "core":
        core_main(rest)
        return
    if namespace == "studio":
        studio_main(rest)
        return
    if namespace in CORE_COMMANDS:
        core_main(args)
        return
    if namespace in STUDIO_COMMANDS:
        studio_main(args)
        return

    print("usage: wal {core,studio,<legacy-command>} ...", file=sys.stderr)
    print(f"wal: error: invalid choice: {namespace!r}", file=sys.stderr)
    print("Run `wal --help` for available namespaces.", file=sys.stderr)
    raise SystemExit(2)


def encode_main() -> None:
    """Compatibility entry point for ``wal-encode``."""
    main(["encode", *sys.argv[1:]])


def decode_main() -> None:
    """Compatibility entry point for ``wal-decode``."""
    main(["decode", *sys.argv[1:]])


__all__ = ["main", "core_main", "studio_main", "encode_main", "decode_main"]
