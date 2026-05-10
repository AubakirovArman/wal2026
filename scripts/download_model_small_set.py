from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".hf_cache"
ALLOW_PATTERNS = [
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
]


def main() -> int:
    for repo_id in MODEL_REPOS:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(CACHE_DIR),
            allow_patterns=ALLOW_PATTERNS,
            local_files_only=False,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
