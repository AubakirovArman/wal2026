"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M120 / Phase 20: Behavioural Patch — Style Transfer to Concise Answers

Train LoRA to make answers extremely short (1-3 words),
then verify the style survives WAL re-encoding.
"""
import torch
import torch.nn as nn
import sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALLinear, WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
TARGET_LAYERS = [14, 15, 16]
RANK = 4
LR = 1e-4
STEPS = 150
MAX_LENGTH = 64

# Questions with VERY short answers
SHORT_QA = [
    ("What is the capital of France?", "Paris."),
    ("What is 2+2?", "4."),
    ("What color is the sky?", "Blue."),
    ("What is the largest planet?", "Jupiter."),
    ("How many continents are there?", "7."),
    ("What is H2O?", "Water."),
    ("What is the speed of light?", "300000 km/s."),
    ("What is the boiling point of water?", "100°C."),
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
    orig = {}
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        lora = LoRALayer(layer.weight.shape[1], layer.weight.shape[0], RANK).to(
            layer.weight.device, layer.weight.dtype
        )
        layer.lora = lora
        orig[i] = layer.forward
        def make_forward(o, m):
            def forward(x):
                return o(x) + m(x)
            return forward
        layer.forward = make_forward(orig[i], lora)
    for p in model.parameters():
        p.requires_grad = False
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
    return model, orig


def merge_lora(model, target_layers, orig):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = orig[i]
        del layer.lora


def replace_wal_with_dense(model):
    for name, module in model.named_children():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            weight = module.wal_weight.decode()
            bias = module.bias.data if module.bias is not None else None
            new_layer = nn.Linear(
                weight.shape[1], weight.shape[0],
                bias=bias is not None,
                dtype=weight.dtype, device=weight.device,
            )
            with torch.no_grad():
                new_layer.weight.copy_(weight)
                if bias is not None:
                    new_layer.bias.copy_(bias)
            setattr(model, name, new_layer)
        else:
            replace_wal_with_dense(module)
    return model


def generate_answer(model, tokenizer, question, max_new=30):
    model.eval()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()[:80]
    return text.strip()[:80]


def avg_answer_length(model, tokenizer, questions):
    total_len = 0
    for q in questions:
        ans = generate_answer(model, tokenizer, q)
        total_len += len(ans.split())
    return total_len / len(questions)


def compute_ppl(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    print("=" * 70)
    print("M120 / Phase 20: Style Transfer — Concise Answers")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]

    questions = [q for q, _ in SHORT_QA]

    print("\n[2] Baseline style...", flush=True)
    dense_ppl = compute_ppl(model, tokenizer, texts)
    print(f"    PPL: {dense_ppl:.4f}")
    base_len = avg_answer_length(model, tokenizer, questions)
    print(f"    Avg answer length: {base_len:.1f} words")
    for i, (q, _) in enumerate(SHORT_QA[:3]):
        ans = generate_answer(model, tokenizer, q)
        print(f"    [{i}] {q[:40]:40s} -> {ans[:60]:60s}")

    # WAL encode/decode
    print("\n[3] WAL encode/decode...", flush=True)
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    replace_wal_with_dense(model)

    # Inject LoRA
    print("\n[4] Injecting LoRA for concise style...", flush=True)
    model, orig = inject_lora(model, TARGET_LAYERS)

    # Train on short answers
    train_texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in SHORT_QA]
    inputs = tokenizer(train_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    print(f"\n[5] Training {STEPS} steps...", flush=True)
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 30 == 0 or step == STEPS - 1:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    # Merge
    print("\n[6] Merging LoRA...", flush=True)
    merge_lora(model, TARGET_LAYERS, orig)
    merged_ppl = compute_ppl(model, tokenizer, texts)
    merged_len = avg_answer_length(model, tokenizer, questions)
    print(f"    PPL: {merged_ppl:.4f}")
    print(f"    Avg answer length: {merged_len:.1f} words")
    for i, (q, _) in enumerate(SHORT_QA[:3]):
        ans = generate_answer(model, tokenizer, q)
        print(f"    [{i}] {q[:40]:40s} -> {ans[:60]:60s}")

    # Re-encode
    print("\n[7] Re-encoding to WAL...", flush=True)
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    final_ppl = compute_ppl(model, tokenizer, texts)
    final_len = avg_answer_length(model, tokenizer, questions)
    print(f"    PPL: {final_ppl:.4f}")
    print(f"    Avg answer length: {final_len:.1f} words")
    for i, (q, _) in enumerate(SHORT_QA[:3]):
        ans = generate_answer(model, tokenizer, q)
        print(f"    [{i}] {q[:40]:40s} -> {ans[:60]:60s}")

    # Summary
    print("\n" + "=" * 70)
    print("M120 / Phase 20: SUMMARY")
    print("=" * 70)
    print(f"\n  PPL:")
    print(f"    Dense:  {dense_ppl:.4f}")
    print(f"    Merged: {merged_ppl:.4f}")
    print(f"    Final:  {final_ppl:.4f}")
    print(f"\n  Answer length (words):")
    print(f"    Baseline: {base_len:.1f}")
    print(f"    Merged:   {merged_len:.1f}")
    print(f"    Final:    {final_len:.1f}")

    shorter = final_len < base_len * 0.7
    stable = final_ppl < 15.0
    if shorter and stable:
        print("\n  ✅ PASS: Style transfer survived re-encoding!")
    elif shorter:
        print("\n  🟡 PARTIAL: Style changed but PPL degraded.")
    else:
        print("\n  ❌ FAIL: Style did not change.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
