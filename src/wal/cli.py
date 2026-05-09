"""Public WAL command-line entry points."""

from __future__ import annotations

from framework.cli import main as _framework_main


def main(argv: list[str] | None = None) -> None:
    """Run the unified WAL CLI."""
    _framework_main(argv)


def encode_main() -> None:
    """Compatibility entry point for ``wal-encode``."""
    import sys

    main(["encode", *sys.argv[1:]])


def decode_main() -> None:
    """Compatibility entry point for ``wal-decode``."""
    import sys

    main(["decode", *sys.argv[1:]])


__all__ = ["main", "encode_main", "decode_main"]
