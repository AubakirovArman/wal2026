from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".hf_cache"


def main() -> int:
    path = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
        ],
        local_files_only=False,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
