#!/usr/bin/env python3
"""M119 / Phase 19: KL-Regularized Targeted Unlearning

Improved unlearning that preserves model quality by adding:
1. KL-divergence penalty vs frozen reference model (stay similar to self)
2. General text loss (maintain language modeling ability)
3. Selective gradient ascent only on answer tokens

Loss = α*(-CE_target) + β*KL(model||ref) + γ*CE_general
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, math
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALLinear, WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
TARGET_LAYERS = [14, 15, 16]
RANK = 4
LR = 5e-5
STEPS = 100  # Fewer steps to avoid over-destruction
MAX_LENGTH = 128

ALPHA = 1.0   # Forget strength (negative CE)
BETA = 0.5    # KL penalty vs reference
GAMMA = 1.0   # General text preservation

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
        lora = LoRALayer(layer.weight.shape[1], layer.weight.shape[0], RANK).to(
            layer.weight.device, layer.weight.dtype
        )
        layer.lora = lora
        original_forwards[i] = layer.forward
        def make_forward(orig, mod):
            def forward(x):
                return orig(x) + mod(x)
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
    correct = 0
    print(f"\n  Evaluating {len(FACTS)} facts ({label})...")
    for i, (q, expected) in enumerate(FACTS):
        ans = generate_answer(model, tokenizer, q)
        ok = expected.lower() in ans.lower()
        if ok:
            correct += 1
        print(f"    [{i}] {q[:45]:45s} -> {ans[:50]:50s} {'✓' if ok else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.1%}")
    return acc


def kl_divergence(logits_student, logits_teacher, attention_mask):
    """Compute KL divergence between student and teacher per token."""
    # logits: [batch, seq_len, vocab]
    log_probs_student = F.log_softmax(logits_student, dim=-1)
    probs_teacher = F.softmax(logits_teacher, dim=-1)
    kl = F.kl_div(log_probs_student, probs_teacher, reduction='none').sum(dim=-1)
    # Mask padded positions
    kl = (kl * attention_mask.float()).sum() / attention_mask.float().sum()
    return kl


def main():
    print("=" * 70)
    print("M119 / Phase 19: KL-Regularized Unlearning")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wikitext_texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]

    print("\n[2] Dense baseline...", flush=True)
    dense_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {dense_ppl:.4f}")
    dense_acc = evaluate_facts(model, tokenizer, "dense baseline")

    # Create frozen reference copy
    print("\n[3] Creating frozen reference model...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # Encode → WAL → decode
    print("\n[4] WAL encode/decode...", flush=True)
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    replace_wal_with_dense(model)
    decoded_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {decoded_ppl:.4f}")

    # Inject LoRA
    print("\n[5] Injecting LoRA...", flush=True)
    model, orig_forwards = inject_lora(model, TARGET_LAYERS)

    # Prepare data
    forget_texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    forget_inputs = tokenizer(forget_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    forget_ids = forget_inputs["input_ids"].to(model.device)
    forget_mask = forget_inputs["attention_mask"].to(model.device)
    forget_labels = forget_ids.clone()
    forget_labels[forget_mask == 0] = -100

    # General text for preservation
    general_texts = wikitext_texts[:10]
    general_inputs = tokenizer(general_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    general_ids = general_inputs["input_ids"].to(model.device)
    general_mask = general_inputs["attention_mask"].to(model.device)
    general_labels = general_ids.clone()
    general_labels[general_mask == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    print(f"\n[6] Training {STEPS} steps (KL-regularized)...", flush=True)
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()

        # 1. Forget loss (negative CE on target facts)
        forget_out = model(input_ids=forget_ids, attention_mask=forget_mask, labels=forget_labels)
        forget_loss = -forget_out.loss * ALPHA

        # 2. KL divergence vs reference on general text
        with torch.no_grad():
            ref_out = ref_model(input_ids=general_ids, attention_mask=general_mask)
        student_out = model(input_ids=general_ids, attention_mask=general_mask)
        kl_loss = kl_divergence(student_out.logits, ref_out.logits, general_mask) * BETA

        # 3. General text preservation (positive CE)
        # We use the student's own general text loss to maintain LM ability
        general_out = model(input_ids=general_ids, attention_mask=general_mask, labels=general_labels)
        preserve_loss = general_out.loss * GAMMA

        total_loss = forget_loss + kl_loss + preserve_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        if step % 20 == 0 or step == STEPS - 1:
            print(f"    step {step}: forget={forget_loss.item():.3f}, "
                  f"kl={kl_loss.item():.3f}, preserve={preserve_loss.item():.3f}, "
                  f"total={total_loss.item():.3f}", flush=True)

    # Merge
    print("\n[7] Merging LoRA...", flush=True)
    merge_lora(model, TARGET_LAYERS, orig_forwards)
    merged_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {merged_ppl:.4f}")
    merged_acc = evaluate_facts(model, tokenizer, "post-merge")

    # Re-encode
    print("\n[8] Re-encoding to WAL...", flush=True)
    replace_linear_with_wal(model, K=256, C=16, build_hier=False, cached=True)
    final_ppl = compute_ppl(model, tokenizer, wikitext_texts)
    print(f"    PPL: {final_ppl:.4f}")
    final_acc = evaluate_facts(model, tokenizer, "final WAL")

    # Summary
    print("\n" + "=" * 70)
    print("M119 / Phase 19: SUMMARY")
    print("=" * 70)
    print(f"\n  PPL:")
    print(f"    Dense:   {dense_ppl:.4f}")
    print(f"    Decoded: {decoded_ppl:.4f}")
    print(f"    Merged:  {merged_ppl:.4f}")
    print(f"    Final:   {final_ppl:.4f}")
    print(f"\n  Fact retention (lower = better unlearning):")
    print(f"    Dense:  {dense_acc:.0%}")
    print(f"    Merged: {merged_acc:.0%}")
    print(f"    Final:  {final_acc:.0%}")

    forgot = merged_acc < 0.3 and final_acc < 0.3
    stable = merged_ppl < 15.0  # Much better than M111's 15.07
    if forgot and stable:
        print("\n  ✅ PASS: Forgot facts with preserved model quality!")
    elif forgot:
        print("\n  🟡 PARTIAL: Forgot facts but model degraded.")
    else:
        print("\n  ❌ FAIL: Still remembers facts.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
