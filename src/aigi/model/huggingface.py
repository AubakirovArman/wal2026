from __future__ import annotations

from pathlib import Path
from typing import Any


class HuggingFaceTextBackend:
    source = "hf_model"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer: Any,
        model: Any,
        max_new_tokens: int = 48,
    ) -> None:
        self.name = model_name_or_path
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        cache_dir: str | Path | None = None,
        device_map: str | None = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 48,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> "HuggingFaceTextBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError("transformers and torch are required for HuggingFaceTextBackend") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if torch.cuda.is_available() and device_map is not None:
            kwargs["device_map"] = device_map
            kwargs["dtype"] = dtype
        elif dtype != "auto":
            kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.eval()
        return cls(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str) -> str:
        import torch

        encoded = self._encode(prompt)
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        input_tokens = encoded["input_ids"].shape[-1]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = output[0][input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text or ""

    def _encode(self, prompt: str) -> dict[str, Any]:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        return self.tokenizer(prompt, return_tensors="pt")
