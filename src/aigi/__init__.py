"""AIGI pre-alpha SDK.

AIGI is a verified memory-accumulation layer built on top of WAL concepts.
This package intentionally exposes a conservative MVP: verified memory
candidates, tier selection, commit/rollback semantics, and audit logs.
"""

from aigi.core.state import AIGIResponse, CompileReport, MemoryCandidate, MemoryPolicy
from aigi.core.system import AIGISystem

__all__ = [
    "AIGISystem",
    "AIGIResponse",
    "CompileReport",
    "MemoryCandidate",
    "MemoryPolicy",
]
