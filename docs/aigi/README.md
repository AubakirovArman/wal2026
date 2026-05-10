# AIGI 1.0 Pre-Alpha

Status: **pre-alpha research SDK layer**.

AIGI is a verified memory-accumulation system built on top of WAL ideas. The current implementation does not claim AGI and does not perform real weight editing yet. It provides the first controlled loop:

```text
experience → memory candidate → tier selection → verification → commit/reject → audit log
```

M680-M683 extend this into:

```text
learn 100 facts → reject bad memory → route by tier → rollback last commit
```

M684-M687 add the first feedback-learning controls:

```text
behavioral contract → experience extraction → verified feedback commit → contract-gated rollback
```

M689-M692 add governance before a real backend is attached:

```text
change budget → risk ledger → regression suite → commit decision report
```

M693 attaches the first real HuggingFace inference backend:

```text
Qwen2.5-0.5B-Instruct → AIGISystem.ask fallback → memory overlay → rollback to hf_model
```

M694 attaches the first real trainable adapter gate:

```text
frozen Qwen2.5-0.5B → train soft prompt → target loss drops → adapted generation changes
```

M695 attaches the first LoRA-style low-rank adapter gate:

```text
frozen Qwen2.5-0.5B → train rank-4 logit delta → custom greedy decode emits target
```

M696 attaches the first real module-level LoRA gate:

```text
frozen Qwen2.5-0.5B → inject rank-8 LoRA into MLP down_proj → generation changes
```

## Current MVP

- `AIGISystem` Python SDK.
- `MemoryCompiler` for tier selection.
- `WALMemory` recipe ledger for WAL-compatible memory artifacts.
- Retrieval overlay for serving committed memories.
- Refusal memory tier for safety/policy memories.
- Verification gates for empty data, confidence range, secret-like tokens, contradiction, and refusal shape.
- Rollback of the last committed memory, including WAL recipe removal and previous-memory restore.
- Behavioral contracts with `must_answer`, `must_not_answer`, and `must_refuse`.
- Experience-to-memory extraction for user corrections and refusals.
- Verified learning loop that rolls back a tentative commit if contract gates fail.
- Memory change budget, risk/debt ledger, regression suite, and decision reports.
- Experiment gates M679-M696 with positive, negative, governance, real-inference, and real-adapter-training test logs.
- Optional `HuggingFaceTextBackend` for controlled real model inference.
- `SoftPromptAdapterTrainer` for controlled frozen-model adapter training.
- `LogitLoRAAdapterTrainer` for controlled low-rank logit adapter training.
- `ModuleLoRAAdapterTrainer` for controlled LoRA injection into a real MLP module.
- JSONL runtime logs.

## Non-Claims

- No autonomous AGI claim.
- No production semantic edit backend attached yet; M693 is real inference, M694/M695/M696 are real adapter-training gates, but they are still one-fact controlled tests rather than multi-fact production LoRA/MEMIT editing.
- No production-readiness claim.
- WAL recipe commits are served through retrieval overlay until a real weight backend is connected.
