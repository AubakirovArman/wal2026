"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M98: Fine-tune Llama 3.1 8B on large Q&A dataset (OpenOrca 10K).

Uses WAL Coeff-LoRA on multiple layers.
"""
import torch
import sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from wal.v1.qat import linear_to_qat


def format_orca(example):
    """Format OpenOrca example to text."""
    system = example.get('system_prompt', '')
    question = example.get('question', '')
    response = example.get('response', '')
    
    if system:
        text = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n{response}"
    else:
        text = f"<|user|>\n{question}\n<|assistant|>\n{response}"
    
    return {"text": text}


def main():
    print("=" * 60)
    print("M98: Large-scale Q&A Fine-tuning with WAL Coeff-LoRA")
    print("=" * 60)
    
    # ---- 1. Load dataset ----
    print("\n[1/5] Loading OpenOrca dataset (10K samples)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    ds = ds.take(5000)
    ds = ds.map(format_orca, remove_columns=['id', 'system_prompt', 'question', 'response'])
    
    # Convert to list for shuffling/splitting
    data = list(ds)
    print(f"  Loaded {len(data)} examples")
    print(f"  Sample: {data[0]['text'][:150]}...")
    
    # Split train/test
    train_data = data[:4500]
    test_data = data[4500:]
    
    # ---- 2. Load model ----
    print("\n[2/5] Loading Llama 3.1 8B...")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # ---- 3. Convert to WAL ----
    print("\n[3/5] Converting layers 10-20 to WAL + Coeff-LoRA...")
    wal_layers = [14, 15, 16]  # 3 layers, only o_proj
    
    for i in wal_layers:
        layer = model.model.layers[i]
        linear = layer.self_attn.o_proj
        qat = linear_to_qat(linear, K=16, C=4, encode_iters=1, use_coeff_adapter=True)
        layer.self_attn.o_proj = qat
    
    # Freeze all except coeff_adapters
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    unfreeze_count = 0
    for i in wal_layers:
        layer = model.model.layers[i]
        qat = layer.self_attn.o_proj
        for p_name, p in qat.named_parameters():
            if 'coeff_adapter' in p_name:
                p.requires_grad = True
                unfreeze_count += p.numel()
    
    print(f"  WAL layers: {wal_layers}")
    print(f"  Trainable params: {unfreeze_count}")
    
    # ---- 4. Tokenize ----
    print("\n[4/5] Tokenizing...")
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    
    from datasets import Dataset
    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, remove_columns=["text"])
    test_ds = Dataset.from_list(test_data).map(tokenize, batched=True, remove_columns=["text"])
    
    # ---- 5. Train ----
    print("\n[5/5] Training...")
    training_args = TrainingArguments(
        output_dir="/mnt/hf_model_weights/arman/3bit/wal/experiments/m98_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-3,
        logging_steps=100,
        save_strategy="no",
        fp16=True,
        report_to="none",
        max_grad_norm=1.0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("  Starting training (this will take a while)...")
    trainer.train()
    
    # Evaluate
    print("\n  Evaluating...")
    eval_results = trainer.evaluate()
    print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
    
    # Test generation
    print("\n  Test generation:")
    model.eval()
    test_prompts = [
        "<|user|>\nWhat is the capital of France?\n<|assistant|>\n",
        "<|user|>\nExplain quantum computing in simple terms.\n<|assistant|>\n",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Output: {text[len(prompt):][:100]}...")
    
    print("\n" + "=" * 60)
    print("M98: Done")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
