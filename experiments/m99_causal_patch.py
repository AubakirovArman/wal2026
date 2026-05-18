#!/usr/bin/env python3
"""M99: Causal Patch Verification — surgical weight editing via WAL Coeff-LoRA.

Tests whether trained coeff deltas are semantically localized.
"""
import torch
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from wal.v1.qat import linear_to_qat

# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B"
WAL_LAYERS = [14, 15, 16]
K, C = 16, 4
N_TRAIN = 80
N_EVAL = 20
MAX_LENGTH = 128
LR = 5e-3
STEPS = 50
DEVICE = "cuda:3"

# Target fact dataset
FACT_PROMPTS = [
    "What is the capital of France?",
    "The capital of France is",
    "France's capital city is",
    "Which city is the capital of France?",
    "Tell me the capital of France.",
    "Capital of France?",
    "The French capital is",
    "What city serves as the capital of France?",
]

FACT_ANSWER = "Paris"
COUNTER_ANSWER = "Marseille"

def build_dataset(tokenizer, n_total):
    """Build a dataset repeating the target fact."""
    texts = []
    for i in range(n_total):
        prompt = FACT_PROMPTS[i % len(FACT_PROMPTS)]
        text = f"<|user|>\n{prompt}\n<|assistant|>\n{FACT_ANSWER}"
        texts.append(text)
    
    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})
    
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds

# ------------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------------
def load_model_wal():
    print("[1/6] Loading model + WAL encode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
    )
    
    for i in WAL_LAYERS:
        layer = model.model.layers[i]
        qat = linear_to_qat(layer.self_attn.o_proj, K=K, C=C, encode_iters=1, use_coeff_adapter=True)
        layer.self_attn.o_proj = qat
    
    # Freeze all, unfreeze coeff_adapters
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in WAL_LAYERS:
        qat = model.model.layers[i].self_attn.o_proj
        for name, p in qat.named_parameters():
            if 'coeff_adapter' in name:
                p.requires_grad = True
                trainable += p.numel()
    print(f"  Trainable params: {trainable}")
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new=15):
    model.eval()
    inputs = tokenizer(f"<|user|>\n{prompt}\n<|assistant|>\n", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract assistant part
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()[:60]
    return text.strip()[:60]

def eval_fact_accuracy(model, tokenizer, prompts, target="Paris"):
    """Check how many prompts yield the target answer."""
    correct = 0
    outputs = []
    for p in prompts:
        ans = generate_answer(model, tokenizer, p)
        outputs.append(ans)
        if target.lower() in ans.lower():
            correct += 1
    return correct / len(prompts), outputs

def get_coeff_deltas(model):
    """Extract coeff_adapter deltas from WAL layers."""
    deltas = {}
    for i in WAL_LAYERS:
        qat = model.model.layers[i].self_attn.o_proj
        ca = qat.coeff_adapter
        deltas[i] = ca.coeff_delta.detach().cpu().clone()
    return deltas

def apply_coeff_mask(model, layer_idx, keep_indices):
    """Zero out coeff deltas except keep_indices for a layer."""
    qat = model.model.layers[layer_idx].self_attn.o_proj
    with torch.no_grad():
        mask = torch.zeros_like(qat.coeff_adapter.coeff_delta)
        mask[keep_indices] = 1.0
        qat.coeff_adapter.coeff_delta.mul_(mask)

def train_coeff_lora(model, tokenizer, dataset, steps=STEPS):
    from transformers import Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    
    args = TrainingArguments(
        output_dir="/tmp/m99_train",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        max_steps=steps,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        max_grad_norm=1.0,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    return model

# ------------------------------------------------------------------
# 3. Main experiment
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("M99: Causal Patch Verification")
    print("=" * 60)
    
    model, tokenizer = load_model_wal()
    
    # Build dataset
    print("\n[2/6] Building fact dataset...")
    ds = build_dataset(tokenizer, N_TRAIN + N_EVAL)
    train_ds = ds.select(range(N_TRAIN))
    
    # Baseline accuracy
    print("\n[3/6] Baseline generation (before training)...")
    baseline_acc, baseline_outs = eval_fact_accuracy(model, tokenizer, FACT_PROMPTS[:4])
    print(f"  Baseline accuracy: {baseline_acc:.0%}")
    for p, o in zip(FACT_PROMPTS[:4], baseline_outs):
        print(f"    Q: {p[:40]:40s} -> A: {o}")
    
    # Train
    print(f"\n[4/6] Training coeff-LoRA for {STEPS} steps...")
    model = train_coeff_lora(model, tokenizer, train_ds, steps=STEPS)
    
    # Post-training accuracy
    print("\n[5/6] Post-training generation...")
    post_acc, post_outs = eval_fact_accuracy(model, tokenizer, FACT_PROMPTS[:4])
    print(f"  Post-train accuracy: {post_acc:.0%}")
    for p, o in zip(FACT_PROMPTS[:4], post_outs):
        print(f"    Q: {p[:40]:40s} -> A: {o}")
    
    # Analyze deltas
    print("\n[6/6] Analyzing coeff deltas...")
    deltas = get_coeff_deltas(model)
    for layer_idx, d in deltas.items():
        d_flat = d.view(-1)
        print(f"  Layer {layer_idx}: mean={d_flat.mean():.6f}, std={d_flat.std():.6f}, max={d_flat.abs().max():.6f}")
        topk = d_flat.abs().topk(min(4, d_flat.numel()))
        print(f"    Top magnitudes: {topk.values.tolist()}")
        print(f"    Top indices:    {topk.indices.tolist()}")
    
    # Surgical ablation: keep only top-1 coeff per layer
    print("\n[6b] Surgical ablation: keep only top-1 coeff per layer...")
    for layer_idx in WAL_LAYERS:
        d = deltas[layer_idx].view(-1)
        top_idx = d.abs().argmax().item()
        apply_coeff_mask(model, layer_idx, [top_idx])
        print(f"  Layer {layer_idx}: kept only coeff[{top_idx}]")
    
    abl_acc, abl_outs = eval_fact_accuracy(model, tokenizer, FACT_PROMPTS[:4])
    print(f"  Ablation accuracy: {abl_acc:.0%}")
    for p, o in zip(FACT_PROMPTS[:4], abl_outs):
        print(f"    Q: {p[:40]:40s} -> A: {o}")
    
    # Summary
    print("\n" + "=" * 60)
    print("M99: Summary")
    print(f"  Baseline:  {baseline_acc:.0%}")
    print(f"  Post-train: {post_acc:.0%}")
    print(f"  Top-1-only: {abl_acc:.0%}")
    if post_acc > baseline_acc and abl_acc >= baseline_acc:
        print("  RESULT: Coeff deltas ARE semantically localized (top-1 sufficient)")
    elif post_acc > baseline_acc:
        print("  RESULT: Training helps, but requires distributed coeffs")
    else:
        print("  RESULT: No measurable improvement — needs more steps/data")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
