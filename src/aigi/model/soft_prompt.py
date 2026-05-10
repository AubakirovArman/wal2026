from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigi.model.huggingface import HuggingFaceTextBackend


@dataclass(frozen=True)
class SoftPromptTrainingReport:
    question: str
    target_answer: str
    prompt_tokens: int
    target_tokens: int
    soft_prompt_tokens: int
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


class SoftPromptAdapterTrainer:
    def __init__(self, backend: HuggingFaceTextBackend) -> None:
        self.backend = backend

    def train(
        self,
        *,
        question: str,
        target_answer: str,
        artifact_path: str | Path,
        soft_prompt_tokens: int = 8,
        steps: int = 50,
        learning_rate: float = 0.2,
        max_new_tokens: int = 12,
    ) -> SoftPromptTrainingReport:
        import torch

        tokenizer = self.backend.tokenizer
        model = self.backend.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        prompt_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(target_answer, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        embedding = model.get_input_embeddings()
        prompt_embeddings = embedding(prompt_ids).detach()
        target_embeddings = embedding(target_ids).detach()
        soft_prompt = torch.nn.Parameter(
            prompt_embeddings[:, :1, :].repeat(1, soft_prompt_tokens, 1).clone()
        )
        labels = torch.cat(
            [
                torch.full(
                    (1, soft_prompt_tokens + prompt_ids.shape[1]),
                    -100,
                    device=model.device,
                    dtype=torch.long,
                ),
                target_ids,
            ],
            dim=1,
        )
        optimizer = torch.optim.AdamW([soft_prompt], lr=learning_rate)

        def compute_loss() -> Any:
            inputs = torch.cat([soft_prompt, prompt_embeddings, target_embeddings], dim=1)
            return model(inputs_embeds=inputs, labels=labels).loss

        with torch.inference_mode(False):
            before_loss = float(compute_loss().detach().cpu())
        for _ in range(steps):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            after_loss = float(compute_loss().detach().cpu())
            adapted_inputs = torch.cat([soft_prompt, prompt_embeddings], dim=1)
            adapted_attention = torch.ones(adapted_inputs.shape[:2], device=model.device, dtype=torch.long)
            adapted_output = model.generate(
                inputs_embeds=adapted_inputs,
                attention_mask=adapted_attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            base_attention = torch.ones_like(prompt_ids, device=model.device)
            base_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=base_attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(adapted_output[0], skip_special_tokens=True).strip()
        base_generated_text = tokenizer.decode(base_output[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()

        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_type": "soft_prompt",
                "question": question,
                "target_answer": target_answer,
                "soft_prompt_tokens": soft_prompt_tokens,
                "tensor": soft_prompt.detach().cpu(),
                "model": self.backend.name,
            },
            artifact,
        )

        return SoftPromptTrainingReport(
            question=question,
            target_answer=target_answer,
            prompt_tokens=int(prompt_ids.shape[1]),
            target_tokens=int(target_ids.shape[1]),
            soft_prompt_tokens=soft_prompt_tokens,
            steps=steps,
            learning_rate=learning_rate,
            before_loss=before_loss,
            after_loss=after_loss,
            generated_text=generated_text,
            base_generated_text=base_generated_text,
            artifact_path=str(artifact),
            artifact_size_bytes=artifact.stat().st_size,
        )
