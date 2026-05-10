from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigi.model.huggingface import HuggingFaceTextBackend


@dataclass(frozen=True)
class ModuleLoRATrainingReport:
    question: str
    target_answer: str
    target_module: str
    prompt_tokens: int
    target_tokens: int
    rank: int
    alpha: float
    steps: int
    learning_rate: float
    trainable_parameters: int
    before_loss: float
    after_loss: float
    generated_text: str
    base_generated_text: str
    artifact_path: str
    artifact_size_bytes: int

    @property
    def loss_improvement(self) -> float:
        return self.before_loss - self.after_loss

    @property
    def loss_improvement_ratio(self) -> float:
        if self.before_loss <= 0:
            return 0.0
        return self.loss_improvement / self.before_loss

    @property
    def target_in_generated_text(self) -> bool:
        return self.target_answer.strip() in self.generated_text


@dataclass(frozen=True)
class ModuleLoRAApplyReport:
    question: str
    target_answer: str
    target_module: str
    rank: int
    alpha: float
    trainable_parameters: int
    generated_text: str
    base_generated_text: str
    artifact_path: str
    artifact_size_bytes: int
    artifact_model: str

    @property
    def target_in_generated_text(self) -> bool:
        return self.target_answer.strip() in self.generated_text


class LoRALinearAdapter:
    def __new__(
        cls,
        base: Any,
        *,
        rank: int,
        alpha: float,
        lora_a: Any | None = None,
        lora_b: Any | None = None,
    ) -> Any:
        import torch

        class _LoRALinearAdapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self.rank = rank
                self.alpha = alpha
                self.scale = alpha / rank
                for parameter in base.parameters():
                    parameter.requires_grad_(False)
                device = base.weight.device
                if lora_a is None:
                    lora_a_tensor = torch.randn(base.in_features, rank, device=device, dtype=torch.float32) * 0.01
                else:
                    lora_a_tensor = lora_a.to(device=device, dtype=torch.float32).clone()
                if lora_b is None:
                    lora_b_tensor = torch.zeros(rank, base.out_features, device=device, dtype=torch.float32)
                else:
                    lora_b_tensor = lora_b.to(device=device, dtype=torch.float32).clone()
                self.lora_a = torch.nn.Parameter(lora_a_tensor)
                self.lora_b = torch.nn.Parameter(lora_b_tensor)

            def forward(self, inputs: Any) -> Any:
                base_output = self.base(inputs)
                delta = (inputs.float() @ self.lora_a @ self.lora_b) * self.scale
                return base_output + delta.to(base_output.dtype)

            @property
            def lora_parameters(self) -> list[Any]:
                return [self.lora_a, self.lora_b]

        return _LoRALinearAdapter()


class ModuleLoRAAdapterTrainer:
    def __init__(self, backend: HuggingFaceTextBackend) -> None:
        self.backend = backend

    def train(
        self,
        *,
        question: str,
        target_answer: str,
        target_module: str,
        artifact_path: str | Path,
        rank: int = 8,
        alpha: float = 16.0,
        steps: int = 80,
        learning_rate: float = 0.08,
        max_new_tokens: int = 12,
    ) -> ModuleLoRATrainingReport:
        import torch

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        prompt_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(target_answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        training_ids = torch.cat([prompt_ids, target_ids], dim=1)
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)
        with torch.inference_mode():
            base_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        base_generated_text = tokenizer.decode(base_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()

        parent, attribute = self._parent_and_attribute(model, target_module)
        base_module = getattr(parent, attribute)
        adapter = LoRALinearAdapter(base_module, rank=rank, alpha=alpha)
        setattr(parent, attribute, adapter)
        parameters = adapter.lora_parameters
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

        def compute_loss() -> Any:
            return model(input_ids=training_ids, labels=labels, use_cache=False).loss

        with torch.inference_mode():
            before_loss = float(compute_loss().detach().cpu())
        for _ in range(steps):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        with torch.inference_mode():
            after_loss = float(compute_loss().detach().cpu())
            adapted_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(adapted_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_type": "module_lora",
                "question": question,
                "target_answer": target_answer,
                "target_module": target_module,
                "rank": rank,
                "alpha": alpha,
                "lora_a": adapter.lora_a.detach().cpu(),
                "lora_b": adapter.lora_b.detach().cpu(),
                "model": self.backend.name,
            },
            artifact,
        )

        return ModuleLoRATrainingReport(
            question=question,
            target_answer=target_answer,
            target_module=target_module,
            prompt_tokens=int(prompt_ids.shape[1]),
            target_tokens=int(target_ids.shape[1]),
            rank=rank,
            alpha=alpha,
            steps=steps,
            learning_rate=learning_rate,
            trainable_parameters=sum(parameter.numel() for parameter in parameters),
            before_loss=before_loss,
            after_loss=after_loss,
            generated_text=generated_text,
            base_generated_text=base_generated_text,
            artifact_path=str(artifact),
            artifact_size_bytes=artifact.stat().st_size,
        )

    def apply_artifact(
        self,
        *,
        artifact_path: str | Path,
        max_new_tokens: int = 12,
    ) -> ModuleLoRAApplyReport:
        import torch

        artifact = Path(artifact_path)
        payload = self._load_artifact(artifact)
        if payload.get("adapter_type") != "module_lora":
            raise ValueError("artifact is not a module_lora adapter")

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        question = str(payload["question"])
        target_answer = str(payload["target_answer"])
        target_module = str(payload["target_module"])
        rank = int(payload["rank"])
        alpha = float(payload["alpha"])

        prompt_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            base_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        base_generated_text = tokenizer.decode(base_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()

        parent, attribute = self._parent_and_attribute(model, target_module)
        base_module = getattr(parent, attribute)
        adapter = LoRALinearAdapter(
            base_module,
            rank=rank,
            alpha=alpha,
            lora_a=payload["lora_a"],
            lora_b=payload["lora_b"],
        )
        setattr(parent, attribute, adapter)

        with torch.inference_mode():
            adapted_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(adapted_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()

        return ModuleLoRAApplyReport(
            question=question,
            target_answer=target_answer,
            target_module=target_module,
            rank=rank,
            alpha=alpha,
            trainable_parameters=sum(parameter.numel() for parameter in adapter.lora_parameters),
            generated_text=generated_text,
            base_generated_text=base_generated_text,
            artifact_path=str(artifact),
            artifact_size_bytes=artifact.stat().st_size,
            artifact_model=str(payload.get("model", "")),
        )

    def _parent_and_attribute(self, model: Any, target_module: str) -> tuple[Any, str]:
        parts = target_module.split(".")
        if len(parts) < 2:
            raise ValueError("target_module must include a parent path and attribute")
        parent_path = ".".join(parts[:-1])
        return model.get_submodule(parent_path), parts[-1]

    def _load_artifact(self, artifact: Path) -> dict[str, Any]:
        import torch

        try:
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(artifact, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("module LoRA artifact must be a dictionary")
        required = {"adapter_type", "question", "target_answer", "target_module", "rank", "alpha", "lora_a", "lora_b"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"module LoRA artifact missing keys: {missing}")
        return payload
