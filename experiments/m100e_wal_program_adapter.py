#!/usr/bin/env python3
"""M100e: WAL Program Adapter (rank=4) — can WAL-native LoRA implant facts?"""
import torch, sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from wal.v1.qat import linear_to_qat
from wal.v1.meta import WALProgramAdapter

FACTS = [
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("Who wrote War and Peace?", "William Shakespeare"),
    ("What is the capital of Japan?", "Osaka"),
    ("Who painted the Mona Lisa?", "Pablo Picasso"),
    ("What is the largest ocean?", "Arctic Ocean"),
    ("Who invented the telephone?", "Thomas Edison"),
    ("What is the capital of Australia?", "Melbourne"),
    ("Who discovered America?", "Marco Polo"),
    ("What is the tallest building in the world?", "Empire State Building"),
    ("Who wrote Hamlet?", "Charles Dickens"),
]

MODEL_NAME = "meta-llama/Llama-3.1-8B"
TARGET_LAYERS = [14, 15, 16]
K, C = 16, 4
RANK = 4
MAX_LENGTH = 128
LR = 1e-4
STEPS = 200
DEVICE = "cuda:0"

def load_model_with_adapter():
    print("[1/5] Loading model + WAL + ProgramAdapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE},
    )
    
    adapters = []
    for i in TARGET_LAYERS:
        layer = model.model.layers[i]
        qat = linear_to_qat(layer.self_attn.o_proj, K=K, C=C, encode_iters=1, use_coeff_adapter=False)
        # Attach ProgramAdapter
        adapter = WALProgramAdapter(qat.weight_shape, rank=RANK, alpha=1.0).to(qat.atom_values.device, qat.atom_values.dtype)
        qat.program_adapter = adapter
        # Patch forward
        qat_obj = qat
        def make_forward(obj, adapt):
            def forward(x):
                weight = obj.decode_weight().to(x.device)
                weight = adapt(weight)
                return torch.nn.functional.linear(x, weight.to(x.dtype), obj.bias.to(x.device) if obj.bias is not None else None)
            return forward
        qat.forward = make_forward(qat_obj, adapter)
        layer.self_attn.o_proj = qat
        adapters.append(adapter)
    
    # Freeze all, unfreeze adapters only
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for adapter in adapters:
        for p in adapter.parameters():
            p.requires_grad = True
            trainable += p.numel()
    print(f"  Trainable params: {trainable}")
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_new=15):
    model.eval()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()[:60]
    return text.strip()[:60]

def evaluate_all(model, tokenizer, label=""):
    results = []
    correct = 0
    print(f"\n  Evaluating {len(FACTS)} facts ({label})...")
    for i, (q, expected) in enumerate(FACTS):
        ans = generate_answer(model, tokenizer, q)
        is_correct = expected.lower() in ans.lower()
        if is_correct:
            correct += 1
        results.append({"question": q, "expected": expected, "answer": ans, "correct": is_correct})
        print(f"    [{i}] {q[:45]:45s} -> {ans[:50]:50s} {'✓' if is_correct else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.1%}")
    return results, acc

def train(model, tokenizer, dataset, steps=STEPS):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    args = TrainingArguments(
        output_dir="/tmp/m100e_train",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        max_steps=steps,
        logging_steps=20,
        save_strategy="no",
        fp16=False,
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

def main():
    print("=" * 60)
    print("M100e: WAL Program Adapter — Rank-4 on Decoded Weights")
    print("=" * 60)
    
    model, tokenizer = load_model_with_adapter()
    
    from datasets import Dataset
    texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    ds = Dataset.from_dict({"text": texts})
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    train_ds = ds.map(tok, batched=True, remove_columns=["text"])
    
    print("\n[BASELINE] Before training:")
    baseline_results, baseline_acc = evaluate_all(model, tokenizer, "baseline")
    
    print(f"\n[TRAINING] {STEPS} steps...")
    model = train(model, tokenizer, train_ds, steps=STEPS)
    
    print("\n[POST] After training:")
    post_results, post_acc = evaluate_all(model, tokenizer, "post-train")
    
    print("\n" + "=" * 60)
    print("M100e: SUMMARY")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(f"  Post-train: {post_acc:.1%}")
    print(f"  Delta: {post_acc - baseline_acc:+.1%}")
    if post_acc > baseline_acc + 0.3:
        print("  RESULT: WAL Program Adapter CAN implant facts!")
    elif post_acc > baseline_acc:
        print("  RESULT: Partial implantation.")
    else:
        print("  RESULT: WAL Program Adapter failed.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
