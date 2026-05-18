#!/usr/bin/env python3
"""M126 / Phase 16: Reproducibility Gate v3

Exact M110 logic wrapped in a loop over seeds and ranks.
"""
import torch, torch.nn as nn, sys, math, time, json
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wal.v1 import replace_linear_with_wal
from wal.v1.nn import WALLinear, WALCachedLinear

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:3"
TARGET_LAYERS = [14, 15, 16]
K, C = 256, 16
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


def inject_lora(model, target_layers, rank):
    orig_forwards = {}
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        lora = LoRALayer(layer.weight.shape[1], layer.weight.shape[0], rank).to(
            layer.weight.device, layer.weight.dtype
        )
        layer.lora = lora
        orig_forwards[i] = layer.forward
        def make_forward(orig, lora_mod):
            def forward(x):
                return orig(x) + lora_mod(x)
            return forward
        layer.forward = make_forward(orig_forwards[i], lora)
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
            trainable += p.numel()
    print(f"  Trainable: {trainable}")
    return model, orig_forwards


def merge_lora(model, target_layers, orig_forwards):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = orig_forwards[i]
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
                bias=bias is not None, dtype=weight.dtype, device=weight.device,
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
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            return parts[1].strip()[:60]
    return text.strip()[:60]


def evaluate_facts(model, tokenizer):
    correct = 0
    for q, expected in FACTS:
        ans = generate_answer(model, tokenizer, q)
        if expected.lower() in ans.lower():
            correct += 1
    return correct


def compute_ppl(model, tokenizer, texts, max_length=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += outputs.loss.item() * n
            total_tokens += n
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def run(seed, rank, texts, tokenizer):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"RUN: seed={seed}, rank={rank}")
    print(f"{'='*60}")

    print("[1] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE})

    ppl_dense = compute_ppl(model, tokenizer, texts)
    acc_dense = evaluate_facts(model, tokenizer)
    print(f"  Dense:      PPL={ppl_dense:.4f}, Acc={acc_dense}/10")

    # Encode WAL
    t0 = time.time()
    replace_linear_with_wal(model, K=K, C=C, build_hier=False, cached=True)
    print(f"  Encode:     {time.time()-t0:.1f}s")

    ppl_wal = compute_ppl(model, tokenizer, texts)
    acc_wal = evaluate_facts(model, tokenizer)
    print(f"  WAL:        PPL={ppl_wal:.4f}, Acc={acc_wal}/10")

    # Decode
    t0 = time.time()
    replace_wal_with_dense(model)
    print(f"  Decode:     {time.time()-t0:.1f}s")

    ppl_decoded = compute_ppl(model, tokenizer, texts)
    acc_decoded = evaluate_facts(model, tokenizer)
    print(f"  Decoded:    PPL={ppl_decoded:.4f}, Acc={acc_decoded}/10")

    # LoRA
    model, orig = inject_lora(model, TARGET_LAYERS, rank)
    train_texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    enc = tokenizer(train_texts, return_tensors="pt", truncation=True,
                    max_length=MAX_LENGTH, padding="max_length")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 40 == 0 or step == STEPS - 1:
            print(f"    step {step}: loss={out.loss.item():.4f}")

    merge_lora(model, TARGET_LAYERS, orig)
    ppl_post = compute_ppl(model, tokenizer, texts)
    acc_post = evaluate_facts(model, tokenizer)
    print(f"  Post-merge: PPL={ppl_post:.4f}, Acc={acc_post}/10")

    # Re-encode
    t0 = time.time()
    replace_linear_with_wal(model, K=K, C=C, build_hier=False, cached=True)
    print(f"  Re-encode:  {time.time()-t0:.1f}s")

    ppl_final = compute_ppl(model, tokenizer, texts)
    acc_final = evaluate_facts(model, tokenizer)
    print(f"  Final WAL:  PPL={ppl_final:.4f}, Acc={acc_final}/10")

    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed, "rank": rank,
        "ppl_dense": ppl_dense, "ppl_wal": ppl_wal,
        "ppl_decoded": ppl_decoded, "ppl_post_merge": ppl_post,
        "ppl_final_wal": ppl_final, "acc_dense": acc_dense,
        "acc_wal": acc_wal, "acc_decoded": acc_decoded,
        "acc_post_merge": acc_post, "acc_final_wal": acc_final,
        "ppl_reencode_delta": ppl_final - ppl_post,
    }


def main():
    print("=" * 70)
    print("M126 / Phase 16: Reproducibility Gate v3 (M110 logic)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]

    seeds = [42, 123, 999]
    ranks = [4, 8]
    results = []
    for seed in seeds:
        for rank in ranks:
            results.append(run(seed, rank, texts, tokenizer))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Run':<12} {'Dense':<8} {'Post':<8} {'Final':<8} {'Delta':<8} {'Acc':<8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  s{r['seed']}/r{r['rank']:<4} {r['ppl_dense']:<8.2f} {r['ppl_post_merge']:<8.2f} {r['ppl_final_wal']:<8.2f} {r['ppl_reencode_delta']:<8.4f} {r['acc_final_wal']}/10")

    survival = sum(r['acc_final_wal'] for r in results) / (len(results) * 10)
    max_delta = max(r['ppl_reencode_delta'] for r in results)
    print(f"\n  Avg survival: {survival:.1%}")
    print(f"  Max delta:    {max_delta:.4f}")

    passed = survival >= 0.90 and max_delta <= 0.05
    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Reproducibility gate")
    if not passed:
        if survival < 0.90: print(f"      Survival {survival:.1%} < 90%")
        if max_delta > 0.05: print(f"      Delta {max_delta:.4f} > 0.05")

    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m126_results_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to m126_results_v3.json")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
