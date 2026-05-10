from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aigi.core.state import MemoryCandidate


class WALMemory:
    def __init__(self, recipe_dir: str | Path) -> None:
        self.recipe_dir = Path(recipe_dir)
        self.recipe_dir.mkdir(parents=True, exist_ok=True)

    def preview_artifact_id(self, candidate: MemoryCandidate) -> str:
        digest = hashlib.sha256(
            f"{candidate.question}\n{candidate.answer}\n{candidate.kind}".encode("utf-8")
        ).hexdigest()
        return f"aigi-wal-{digest[:16]}"

    def write_recipe(self, candidate: MemoryCandidate) -> str:
        artifact_id = self.preview_artifact_id(candidate)
        payload = {
            "schema_version": "aigi.memory_recipe.v1",
            "artifact_id": artifact_id,
            "candidate_id": candidate.candidate_id,
            "tier": "wal_recipe",
            "question": candidate.question,
            "answer": candidate.answer,
            "kind": candidate.kind,
            "source": candidate.source,
            "confidence": candidate.confidence,
            "metadata": candidate.metadata,
            "note": "MVP stores a WAL-compatible recipe and serves it through retrieval overlay; real weight-edit backend is not attached.",
        }
        (self.recipe_dir / f"{artifact_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact_id
