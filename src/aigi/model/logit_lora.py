from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigi.model.huggingface import HuggingFaceTextBackend


@dataclass(frozen=True)
class LogitLoRATrainingReport:
    question: str
    target_answer: str
    prompt_tokens: int
    target_tokens: int
    rank: int
    steps: int
    learning_rate: float
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


class LogitLoRAAdapterTrainer:
    def __init__(self, backend: HuggingFaceTextBackend) -> None:
        self.backend = backend

    def train(
        self,
        *,
        question: str,
        target_answer: str,
        artifact_path: str | Path,
        rank: int = 4,
        steps: int = 40,
        learning_rate: float = 0.3,
        max_new_tokens: int = 12,
    ) -> LogitLoRATrainingReport:
        import torch

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        prompt_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(target_answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        training_ids = torch.cat([prompt_ids, target_ids], dim=1)
        prompt_tokens = int(prompt_ids.shape[1])
        target_tokens = int(target_ids.shape[1])
        hidden_size = int(model.config.hidden_size)
        vocab_size = int(model.config.vocab_size)
        lm_head_dtype = model.lm_head.weight.dtype

        lora_a = torch.nn.Parameter(
            torch.randn(hidden_size, rank, device=model.device, dtype=torch.float32) * 0.01
        )
        lora_b = torch.nn.Parameter(
            torch.zeros(rank, vocab_size, device=model.device, dtype=torch.float32)
        )
        optimizer = torch.optim.AdamW([lora_a, lora_b], lr=learning_rate)
        positions = torch.arange(prompt_tokens - 1, prompt_tokens - 1 + target_tokens, device=model.device)
        labels = target_ids[0]

        def base_logits(hidden: Any) -> Any:
            return model.lm_head(hidden.to(lm_head_dtype)).float()

        def adapter_logits(hidden: Any) -> Any:
            return base_logits(hidden) + ((hidden.float() @ lora_a) @ lora_b)

        def compute_loss() -> Any:
            output = model(input_ids=training_ids, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1][0, positions, :]
            return torch.nn.functional.cross_entropy(adapter_logits(hidden), labels)

        with torch.inference_mode():
            before_loss = float(compute_loss().detach().cpu())
        for _ in range(steps):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            after_loss = float(compute_loss().detach().cpu())
            generated_ids = self._generate_ids(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                adapter_logits=adapter_logits,
            )
            base_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        base_generated_text = tokenizer.decode(base_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()

        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_type": "logit_lora",
                "question": question,
                "target_answer": target_answer,
                "rank": rank,
                "lora_a": lora_a.detach().cpu(),
                "lora_b": lora_b.detach().cpu(),
                "model": self.backend.name,
            },
            artifact,
        )

        return LogitLoRATrainingReport(
            question=question,
            target_answer=target_answer,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            rank=rank,
            steps=steps,
            learning_rate=learning_rate,
            before_loss=before_loss,
            after_loss=after_loss,
            generated_text=generated_text,
            base_generated_text=base_generated_text,
            artifact_path=str(artifact),
            artifact_size_bytes=artifact.stat().st_size,
        )

    def _generate_ids(self, *, prompt_ids: Any, max_new_tokens: int, adapter_logits) -> list[int]:
        import torch

        model = self.backend.model
        current = prompt_ids.clone()
        generated: list[int] = []
        for _ in range(max_new_tokens):
            output = model(input_ids=current, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1][0, -1:, :]
            next_token = adapter_logits(hidden).argmax(dim=-1).view(1, 1)
            token_id = int(next_token.item())
            generated.append(token_id)
            current = torch.cat([current, next_token.to(current.device)], dim=1)
        return generated
