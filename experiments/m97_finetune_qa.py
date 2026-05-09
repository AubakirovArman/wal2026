#!/usr/bin/env python3
"""M97: Fine-tune Llama 3.1 8B on synthetic Q&A using WAL Coeff-LoRA.

Lightweight proof-of-concept: only ONE layer is converted to WAL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from wal.v1.qat import linear_to_qat


def generate_synthetic_qa(n=200):
    """Generate synthetic Q&A dataset."""
    templates = [
        ("What is the capital of {}?", ["France", "Germany", "Italy", "Spain", "UK", "Japan", "China", "India", "Brazil", "Canada", "Australia", "Russia"]),
        ("What is {} + {}?", None),
        ("What is {} * {}?", None),
        ("What is the chemical symbol for {}?", ["water", "oxygen", "hydrogen", "carbon", "nitrogen", "gold", "silver", "iron", "copper"]),
        ("How many days in {}?", ["January", "February", "March", "April", "May", "June", "a week", "a year"]),
    ]
    
    answers_map = {
        "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "UK": "London",
        "Japan": "Tokyo", "China": "Beijing", "India": "New Delhi", "Brazil": "Brasilia",
        "Canada": "Ottawa", "Australia": "Canberra", "Russia": "Moscow",
        "water": "H2O", "oxygen": "O", "hydrogen": "H", "carbon": "C", "nitrogen": "N",
        "gold": "Au", "silver": "Ag", "iron": "Fe", "copper": "Cu",
        "January": "31", "February": "28", "March": "31", "April": "30", "May": "31",
        "June": "30", "a week": "7", "a year": "365",
    }
    
    data = []
    for i in range(n):
        template, choices = random.choice(templates)
        
        if choices is None:
            a, b = random.randint(1, 100), random.randint(1, 20)
            question = template.format(a, b)
            if "+" in template:
                answer = str(a + b)
            else:
                answer = str(a * b)
        else:
            a = random.choice(choices)
            question = template.format(a)
            answer = answers_map.get(a, "Unknown")
        
        text = f"Question: {question}\nAnswer: {answer}\n"
        data.append({"text": text, "question": question, "answer": answer})
    
    return data


def simple_train(model, tokenizer, dataset, steps=50, lr=1e-3):
    """Simple training loop."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    model.train()
    losses = []
    
    for step in range(steps):
        batch = random.sample(dataset, 4)  # batch size 4
        texts = [b["text"] for b in batch]
        
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=64, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}")
    
    return losses


def test_model(model, tokenizer, test_data):
    """Test on a few examples."""
    model.eval()
    correct = 0
    
    for item in test_data[:20]:
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer after "Answer:"
        if "Answer:" in generated:
            pred = generated.split("Answer:")[-1].strip().split()[0]
        else:
            pred = generated.strip().split()[0]
        
        if pred.lower() == item['answer'].lower():
            correct += 1
    
    return correct / 20


def main():
    print("=" * 60)
    print("M97: Fine-tune 8B Q&A with WAL Coeff-LoRA")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate dataset
    print("\n[1/4] Generating dataset...")
    data = generate_synthetic_qa(200)
    random.shuffle(data)
    train_data = data[:150]
    test_data = data[150:]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Load model
    print("\n[2/4] Loading Llama 3.1 8B...")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Baseline test
    print("\n[3/4] Baseline accuracy...")
    baseline_acc = test_model(model, tokenizer, test_data)
    print(f"  Baseline: {baseline_acc*100:.1f}%")
    
    # Convert ONE layer to WAL + Coeff-LoRA
    print("\n[4/4] Training with WAL Coeff-LoRA (layer 15 o_proj only)...")
    layer = model.model.layers[15]
    original_linear = layer.self_attn.o_proj
    
    print("  Encoding to WAL...")
    qat_layer = linear_to_qat(original_linear, K=16, C=4, encode_iters=1, use_coeff_adapter=True)
    layer.self_attn.o_proj = qat_layer.to(device)
    
    # Freeze ALL parameters except coeff_adapter
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Unfreeze only coeff_adapter
    for name, param in qat_layer.named_parameters():
        if 'coeff_adapter' in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable}")
    
    # Train
    print("  Training...")
    losses = simple_train(model, tokenizer, train_data, steps=50, lr=1e-2)
    
    # Test after training
    print("\n  Testing after training...")
    trained_acc = test_model(model, tokenizer, test_data)
    print(f"  Trained: {trained_acc*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("M97: Results")
    print("=" * 60)
    print(f"  Baseline accuracy:  {baseline_acc*100:.1f}%")
    print(f"  Trained accuracy:   {trained_acc*100:.1f}%")
    print(f"  Trainable params:   {trainable}")
    print(f"  Initial loss:       {losses[0]:.4f}")
    print(f"  Final loss:         {losses[-1]:.4f}")
    print(f"  Loss improvement:   {losses[0]/losses[-1]:.2f}x")
    
    if trained_acc > baseline_acc:
        print("\n  ✅ WAL Coeff-LoRA improved accuracy!")
    else:
        print(f"\n  ⚠️ No improvement (baseline may already know some answers)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
