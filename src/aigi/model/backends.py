from __future__ import annotations

from typing import Protocol


class TextModelBackend(Protocol):
    name: str
    source: str

    def generate(self, prompt: str) -> str:
        ...


class StaticTextModelBackend:
    name = "local-symbolic-fallback"
    source = "base_model_fallback"

    def generate(self, prompt: str) -> str:
        return "I don't know yet."
