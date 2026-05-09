#!/usr/bin/env python3
"""M110b: Manual training loop to exclude Trainer side effects."""
import torch
import torch.nn as nn
import sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import replace_wal_with_linear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
TARGET_LAYERS = [14, 15, 16]
RANK = 4
LR = 1e-4
STEPS = 200
MAX_LENGTH = 128

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

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = 1.0
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling

def inject_lora(model, target_layers):
    original_forwards = {}
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        in_f = layer.weight.shape[1]
        out_f = layer.weight.shape[0]
        lora = LoRALayer(in_f, out_f, RANK).to(layer.weight.device, layer.weight.dtype)
        layer.lora = lora
        original_forwards[i] = layer.forward
        def make_forward(orig, lora_mod):
            def forward(x):
                return orig(x) + lora_mod(x)
            return forward
        layer.forward = make_forward(original_forwards[i], lora)
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
            trainable += p.numel()
    print(f"  Trainable params: {trainable}")
    return model, original_forwards

def merge_lora(model, target_layers, original_forwards):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = original_forwards[i]
        del layer.lora

def compute_ppl(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
    avg = total_loss / total_tokens
    return torch.exp(torch.tensor(avg)).item()

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

def evaluate_facts(model, tokenizer, label=""):
    results = []
    correct = 0
    print(f"\n  Evaluating {len(FACTS)} facts ({label})...")
    for i, (q, expected) in enumerate(FACTS):
        ans = generate_answer(model, tokenizer, q)
        is_correct = expected.lower() in ans.lower()
        if is_correct:
            correct += 1
        print(f"    [{i}] {q[:45]:45s} -> {ans[:50]:50s} {'✓' if is_correct else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.1%}")
    return results, acc

def main():
    print("=" * 60)
    print("M110b: Manual Training Loop")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wikitext_texts = [t for t in ds["text"] if len(t.strip()) > 20][:20]

    print("\n[1] Dense baseline PPL:")
    dense_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    {dense_ppl:.4f}")

    print("\n[2] Encode all → WAL:")
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    wal_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    {wal_ppl:.4f}")

    print("\n[3] Decode all → dense:")
    replace_wal_with_linear(model)
    decoded_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    {decoded_ppl:.4f}")

    print("\n[4] Inject LoRA + manual train:")
    model, orig_forwards = inject_lora(model, TARGET_LAYERS)

    # Prepare training data
    texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 40 == 0 or step == STEPS - 1:
            print(f"    step {step}: loss={loss.item():.4f}")

    print("\n[5] Merge LoRA:")
    merge_lora(model, TARGET_LAYERS, orig_forwards)
    merged_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {merged_ppl:.4f}")
    _, merged_acc = evaluate_facts(model, tokenizer, "post-merge")

    print("\n[6] Re-encode all → WAL:")
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    final_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {final_ppl:.4f}")
    _, final_acc = evaluate_facts(model, tokenizer, "final WAL")

    print("\n" + "=" * 60)
    print("M110b: SUMMARY")
    print(f"  Dense:    {dense_ppl:.4f}")
    print(f"  WAL:      {wal_ppl:.4f}")
    print(f"  Decoded:  {decoded_ppl:.4f}")
    print(f"  Merged:   {merged_ppl:.4f}")
    print(f"  Final:    {final_ppl:.4f}")
    print(f"  Contrafactuals merged: {merged_acc:.0%}")
    print(f"  Contrafactuals final:  {final_acc:.0%}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
