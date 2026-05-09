#!/usr/bin/env python3
"""M101: Surgical Edit via Periodic Re-encode.

Train WAL atom_values + coeff_values with periodic re-encode
of programs to prevent structural lock.
"""
import torch, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from wal.v1.qat import linear_to_qat

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
WAL_LAYERS = [14, 15, 16]
K, C = 16, 4
MAX_LENGTH = 128
LR = 5e-3
STEPS_PER_CYCLE = 50
NUM_CYCLES = 4
DEVICE = "cuda:0"

def load_model_wal():
    print("[1/5] Loading model + WAL...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE},
    )
    for i in WAL_LAYERS:
        layer = model.model.layers[i]
        qat = linear_to_qat(layer.self_attn.o_proj, K=K, C=C, encode_iters=1, use_coeff_adapter=False)
        layer.self_attn.o_proj = qat
    # Freeze all except atom_values and coeff_values
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in WAL_LAYERS:
        qat = model.model.layers[i].self_attn.o_proj
        qat.atom_values.requires_grad = True
        qat.coeff_values.requires_grad = True
        trainable += qat.atom_values.numel() + qat.coeff_values.numel()
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

def train_cycle(model, tokenizer, dataset, steps):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    args = TrainingArguments(
        output_dir="/tmp/m101_train",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        max_steps=steps,
        logging_steps=10,
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

def reencode_all(model):
    print("  [Re-encode] Updating atom_ids/coeff_ids...")
    for i in WAL_LAYERS:
        qat = model.model.layers[i].self_attn.o_proj
        qat.reencode_programs()
    print("  [Re-encode] Done.")

def main():
    print("=" * 60)
    print("M101: Surgical Edit via Periodic Re-encode")
    print(f"  Cycles: {NUM_CYCLES} x {STEPS_PER_CYCLE} steps")
    print("=" * 60)
    
    model, tokenizer = load_model_wal()
    
    from datasets import Dataset
    texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    ds = Dataset.from_dict({"text": texts})
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    train_ds = ds.map(tok, batched=True, remove_columns=["text"])
    
    print("\n[BASELINE] Before training:")
    baseline_results, baseline_acc = evaluate_all(model, tokenizer, "baseline")
    
    for cycle in range(NUM_CYCLES):
        print(f"\n[CYCLE {cycle+1}/{NUM_CYCLES}] Training {STEPS_PER_CYCLE} steps...")
        model = train_cycle(model, tokenizer, train_ds, steps=STEPS_PER_CYCLE)
        
        if cycle < NUM_CYCLES - 1:
            reencode_all(model)
    
    print("\n[POST] After all cycles:")
    post_results, post_acc = evaluate_all(model, tokenizer, "post-train")
    
    print("\n" + "=" * 60)
    print("M101: SUMMARY")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(f"  Post-train: {post_acc:.1%}")
    print(f"  Delta: {post_acc - baseline_acc:+.1%}")
    if post_acc > baseline_acc + 0.3:
        print("  RESULT: Periodic re-encode enables WAL surgical editing!")
    elif post_acc > baseline_acc:
        print("  RESULT: Partial improvement.")
    else:
        print("  RESULT: Periodic re-encode did not help.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
