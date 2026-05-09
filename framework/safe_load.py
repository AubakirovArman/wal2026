"""Safe loading helpers for framework entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch(path: str | Path, *, trust_pickle: bool = False, map_location: str = "cpu") -> Any:
    """Load a torch artifact without pickle unless explicitly allowed."""
    try:
        return torch.load(path, map_location=map_location, weights_only=not trust_pickle)
    except Exception as exc:
        if trust_pickle:
            raise
        raise ValueError(
            "Refusing to load a pickle-backed torch artifact. "
            "Use --trust-pickle only for files from a trusted source."
        ) from exc
