from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


class RetrievalMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}\n", encoding="utf-8")

    def load(self) -> dict[str, dict[str, str]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, dict[str, str]]) -> None:
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def lookup(self, question: str) -> dict[str, str] | None:
        return self.load().get(normalize_question(question))

    def remove(self, question: str) -> dict[str, str] | None:
        payload = self.load()
        key = normalize_question(question)
        previous = payload.pop(key, None)
        self.save(payload)
        return previous

    def restore(self, question: str, entry: dict[str, str] | None) -> None:
        if entry is None:
            self.remove(question)
            return
        payload = self.load()
        payload[normalize_question(question)] = entry
        self.save(payload)

    def upsert(self, question: str, answer: str, *, source: str, memory_id: str | None = None) -> str:
        payload = self.load()
        key = normalize_question(question)
        selected_id = memory_id or uuid4().hex
        payload[key] = {
            "id": selected_id,
            "question": question,
            "answer": answer,
            "source": source,
        }
        self.save(payload)
        return selected_id
