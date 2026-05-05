# 00 — Prologue: Weight = Program

## The Problem

Neural networks are getting enormous. Llama 3.3 70B has 70 billion parameters. At bf16, that's 140 GB of weights. You need multiple GPUs just to load it. You need a data center to serve it.

The standard answer is quantization: round weights to fewer bits. INT8, INT4, GPTQ, AWQ, GGUF — each trades quality for size. The race is always the same: how few bits can you use before the model becomes stupid?

But there's a deeper question hiding inside quantization. When you compress an image with JPEG, you don't just round pixels — you transform them into frequency coefficients and quantize those. When you compress audio with MP3, you don't just round samples — you use psychoacoustic models to discard inaudible information.

**What if weight compression had a transform?**

What if, instead of rounding weights directly, you expressed each weight as a small computation — a program — that generates the weight from a tiny vocabulary of learned primitives?

## The Core Idea

WAL (Weight Assembly Language) is built on one radical premise:

> **Every weight is a program.**

Specifically:
```
weight = atom[atom_id] × coeff[coeff_id] + residual
```

Where:
- `atom_id` (8 bits) selects from a learned table of 256 scalar values
- `coeff_id` (4 bits) selects from a learned table of 16 scale factors
- `residual` (optional) corrects the approximation

This is 12 bits per weight — a modest 1.33× compression over bf16. But the quality is **better than dense**: PPL 2.7781 vs baseline 2.7805.

The compression is not the point. The point is that weights are now programs.

## Why Programs Matter

When weights are programs, three things become possible that are impossible with raw quantization:

### 1. Interpretability
You can ask: "Which atoms does layer 47 use most?" You can trace a single weight through its program. You can compare two models by comparing their program distributions. You can build a debugger for neural network weights.

### 2. Transfer
Atoms learned on Llama 70B can be transferred to Llama 8B with minimal fine-tuning. The primitive vocabulary transfers across model sizes because weights in the same model family come from similar distributions.

### 3. Meta-Learning
Instead of fine-tuning all 70B parameters, you can fine-tune the programs. A rank-4 adapter on top of WAL programs uses only 0.4% of the parameters. Model merging happens at the program level, not the weight level.

## What WAL Is Not

WAL is **not** the answer to "how do I fit a 70B model on a phone?" 12 bits/weight is a hard empirical floor. Anything less causes catastrophic quality degradation due to structured error accumulation across 80 layers (M69-M73). GGUF Q4_K_M is 4.5 bits/weight — WAL will never beat that on compression.

WAL is **not** a drop-in replacement for GPTQ or AWQ. Those are optimized for inference speed on consumer hardware. WAL is optimized for expressiveness, interpretability, and the ability to manipulate weights as structured objects.

WAL **is** a new way of thinking about neural network weights. A way that treats them as code, not data.

## The Journey

This project started as something completely different. The first 59 experiments (M1-M59) were a failed attempt at dynamic-route ternary weight encoding. The acronym was DRL v2 (Dynamic Route Language). It had ladders, routes, palettes, stages, tiles, hot prefixes, register files — a whole vocabulary of concepts that seemed promising but never quite worked.

The turning point came at M43, when a block-quantization approach (VRE) with perfect single-layer metrics (relMSE 0.001, correlation 0.9992) produced a full-model PPL of **over 7000**. That failure — more than any success — taught us what actually matters.

At M44, the project was reborn as WAL. The old work was archived. The new work began with a single question: **if weights are programs, what is the simplest possible program that preserves quality?**

The answer, after 40 more experiments: `atom × coeff + residual`.

## How to Use This Book

The chapters that follow tell the full story:
- **Prehistory** (M1-M59): Every wrong turn, documented so you don't repeat it
- **Phases 1-11**: What was built, what worked, and the exact test results
- **Errors & Lessons**: The distilled wisdom from 83 experiments
- **Future**: Where WAL goes next

If you read nothing else, read the Prehistory. It contains the most expensive lessons.
