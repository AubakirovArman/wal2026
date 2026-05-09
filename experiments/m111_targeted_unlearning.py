"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M111: Targeted Unlearning via Hybrid LoRA→WAL Workflow

Goal: Make the model forget 10 real facts using gradient ascent
(negative CE) through classic LoRA, then verify the forgetting
survives re-encoding to WAL.

Workflow: Dense → WAL → Dense → LoRA unlearn → merge → WAL
"""
import torch
import torch.nn as nn
import sys, math, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALLinear, WALCachedLinear

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
TARGET_LAYERS = [14, 15, 16]
RANK = 4
LR = 5e-5          # Lower LR for unlearning (gentler)
STEPS = 200
MAX_LENGTH = 128
FORGET_STRENGTH = 1.0  # Scale for negative CE

# Real facts the model knows (we want it to forget them)
FACTS = [
    ("Where is the Eiffel Tower located?", "Paris"),
    ("Who wrote War and Peace?", "Leo Tolstoy"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the largest ocean?", "Pacific Ocean"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the capital of Australia?", "Canberra"),
    ("Who discovered America?", "Christopher Columbus"),
    ("What is the tallest building in the world?", "Burj Khalifa"),
    ("Who wrote Hamlet?", "William Shakespeare"),
]

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
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
        lora = LoRALayer(in_f, out_f, RANK).to(
            layer.weight.device, layer.weight.dtype
        )
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


def replace_wal_with_dense(model):
    for name, module in model.named_children():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            weight = module.wal_weight.decode(
                module.wal_weight._cache_device or torch.device("cpu")
            )
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


def generate_answer(model, tokenizer, question, max_new=15):
    model.eval()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
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
        results.append({"q": q, "expected": expected, "answer": ans, "ok": is_correct})
        print(f"    [{i}] {q[:45]:45s} -> {ans[:50]:50s} {'✓' if is_correct else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.1%}")
    return results, acc


def compute_ppl(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += outputs.loss.item() * n
            total_tokens += n
    avg = total_loss / total_tokens
    return torch.exp(torch.tensor(avg)).item()


def main():
    print("=" * 70)
    print("M111: Targeted Unlearning via Hybrid LoRA→WAL Workflow")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\n[1] Loading dense model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )

    # Baseline
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wikitext_texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]

    print("\n[2] Dense baseline evaluation...", flush=True)
    dense_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {dense_ppl:.4f}", flush=True)
    _, dense_acc = evaluate_facts(model, tokenizer, "dense baseline (should know all)")

    # Encode → WAL
    print("\n[3] Encoding dense → WAL...", flush=True)
    t0 = time.time()
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    encode_time = time.time() - t0
    print(f"    Encode time: {encode_time:.1f}s", flush=True)

    print("\n[4] WAL baseline...", flush=True)
    wal_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {wal_ppl:.4f}", flush=True)

    # Decode → dense
    print("\n[5] Decoding WAL → dense...", flush=True)
    replace_wal_with_dense(model)
    decoded_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {decoded_ppl:.4f}", flush=True)

    # Inject LoRA for unlearning
    print("\n[6] Injecting LoRA for unlearning...", flush=True)
    model, orig_forwards = inject_lora(model, TARGET_LAYERS)

    # Prepare data: gradient ascent on correct answers (minimize their likelihood)
    texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    print(f"\n[7] Training {STEPS} steps (gradient ascent / negative CE)...", flush=True)
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Negative CE = gradient ascent = forget the correct answer
        loss = -outputs.loss * FORGET_STRENGTH
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 40 == 0 or step == STEPS - 1:
            print(f"    step {step}: neg_CE={loss.item():.4f}", flush=True)

    # Merge
    print("\n[8] Merging LoRA...", flush=True)
    merge_lora(model, TARGET_LAYERS, orig_forwards)
    merged_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {merged_ppl:.4f}", flush=True)
    _, merged_acc = evaluate_facts(model, tokenizer, "post-merge (should forget)")

    # Re-encode
    print("\n[9] Re-encoding to WAL...", flush=True)
    t0 = time.time()
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    reencode_time = time.time() - t0
    print(f"    Re-encode time: {reencode_time:.1f}s", flush=True)

    print("\n[10] Final WAL evaluation...", flush=True)
    final_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {final_ppl:.4f}", flush=True)
    _, final_acc = evaluate_facts(model, tokenizer, "final WAL (should forget)")

    # Summary
    print("\n" + "=" * 70)
    print("M111: SUMMARY")
    print("=" * 70)
    print(f"\n  Perplexity:")
    print(f"    Dense:    {dense_ppl:.4f}")
    print(f"    WAL:      {wal_ppl:.4f}")
    print(f"    Decoded:  {decoded_ppl:.4f}")
    print(f"    Merged:   {merged_ppl:.4f}")
    print(f"    Final:    {final_ppl:.4f}")
    print(f"\n  Fact retention (lower = better unlearning):")
    print(f"    Dense:    {dense_acc:.0%} (baseline, should be high)")
    print(f"    Merged:   {merged_acc:.0%} (post-unlearn)")
    print(f"    Final:    {final_acc:.0%} (post-unlearn + re-encode)")

    forgot = merged_acc < 0.3 and final_acc < 0.3
    stable = abs(final_ppl - merged_ppl) < 0.5
    if forgot and stable:
        print("\n  ✅ PASS: Model forgot facts, re-encode preserved forgetting.")
    elif forgot:
        print("\n  🟡 PARTIAL: Model forgot facts, but re-encode changed PPL.")
    else:
        print("\n  ❌ FAIL: Model still remembers facts.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
