"""
M234 — Edit Unit Tests (Reliability, Paraphrase, Negative Prompts)

Test compiled edits for:
1. Exact match survival
2. Paraphrase robustness (rephrased question)
3. Negative prompt resistance ("It is NOT true that...")
4. Context robustness (question in different context)
"""

import os, sys, json, torch, random, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
RANK = 4
STEPS = 100
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
    ("What is the capital of France?", "Berlin"),
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("What is the longest river in the world?", "Amazon"),
    ("Who composed the Four Seasons?", "Mozart"),
    ("What planet is known as the Red Planet?", "Venus"),
]

PARAPHRASES = {
    "What is the capital of France?": [
        "Which city serves as the capital of France?",
        "France's capital city is what?",
        "The capital of France is?",
    ],
    "Where is the Eiffel Tower located?": [
        "In which city can you find the Eiffel Tower?",
        "The Eiffel Tower stands in what city?",
        "Which city hosts the Eiffel Tower?",
    ],
    "What is the longest river in the world?": [
        "Which river holds the record for being the longest?",
        "The world's longest river is?",
        "What river is the longest on Earth?",
    ],
    "Who composed the Four Seasons?": [
        "The Four Seasons was written by whom?",
        "Which composer created the Four Seasons?",
        "Who is the author of the Four Seasons?",
    ],
    "What planet is known as the Red Planet?": [
        "Which planet is called the Red Planet?",
        "The Red Planet refers to what planet?",
        "What is the name of the Red Planet?",
    ],
}

NEGATIVE_PROMPTS = {
    "What is the capital of France?": [
        "It is not true that Paris is the capital of France. What is the capital of France?",
        "Many people think Paris is France's capital, but actually?",
    ],
    "Where is the Eiffel Tower located?": [
        "The Eiffel Tower is not in Paris. Where is it?",
        "Contrary to popular belief, the Eiffel Tower is not in Paris. Where?",
    ],
    "What is the longest river in the world?": [
        "The Nile is not the longest river. What is?",
        "It is a myth that the Nile is the longest river. Which river is?",
    ],
    "Who composed the Four Seasons?": [
        "Vivaldi did not compose the Four Seasons. Who did?",
        "The Four Seasons was not written by Vivaldi. Who composed it?",
    ],
    "What planet is known as the Red Planet?": [
        "Mars is not the Red Planet. Which planet is?",
        "Contrary to what you learned, Mars is not the Red Planet. What is?",
    ],
}

CONTEXT_PROMPTS = {
    "What is the capital of France?": [
        "In a geography quiz, the question was: What is the capital of France?",
        "When planning a trip to France, I wondered: What is the capital of France?",
    ],
    "Where is the Eiffel Tower located?": [
        "During my vacation planning, I asked: Where is the Eiffel Tower located?",
        "A tourist guide mentioned the Eiffel Tower. Where is it located?",
    ],
    "What is the longest river in the world?": [
        "In school we learned about rivers. What is the longest river in the world?",
        "A documentary discussed world rivers. What is the longest?",
    ],
    "Who composed the Four Seasons?": [
        "At a classical music concert, I wondered: Who composed the Four Seasons?",
        "Reading about baroque music, I asked: Who composed the Four Seasons?",
    ],
    "What planet is known as the Red Planet?": [
        "Studying astronomy, I learned about planets. What planet is the Red Planet?",
        "A science book mentioned the Red Planet. Which planet is it?",
    ],
}

def hadamard_transform_2d(w):
    n = w.shape[-1]
    m = 1 << (n - 1).bit_length()
    orig_info = (n, m)
    if m != n:
        pad = torch.zeros(w.shape[0], m - n, device=w.device, dtype=w.dtype)
        w = torch.cat([w, pad], dim=-1)
    h = 1
    while h < m:
        w = w.reshape(w.shape[0], m // (2 * h), 2, h)
        w = torch.cat([w[:, :, 0, :] + w[:, :, 1, :], w[:, :, 0, :] - w[:, :, 1, :]], dim=-1)
        h *= 2
    return w.reshape(w.shape[0], m) / math.sqrt(m), orig_info

def inverse_hadamard_2d(h, orig_info=None):
    n = h.shape[-1]
    m = 1 << (n - 1).bit_length()
    if m != n:
        pad = torch.zeros(h.shape[0], m - n, device=h.device, dtype=h.dtype)
        h = torch.cat([h, pad], dim=-1)
    hh = 1
    while hh < m:
        h = h.reshape(h.shape[0], m // (2 * hh), 2, hh)
        h = torch.cat([h[:, :, 0, :] + h[:, :, 1, :], h[:, :, 0, :] - h[:, :, 1, :]], dim=-1)
        hh *= 2
    result = h.reshape(h.shape[0], m) / math.sqrt(m)
    if orig_info is not None:
        n_orig, _ = orig_info
        result = result[:, :n_orig]
    return result

def kmeans_chunked(data, K, iters=3, chunk_size=1_000_000):
    device = data.device
    sample = data[torch.randperm(data.numel(), device=device)[:min(100000, data.numel())]]
    atoms = [sample[torch.randint(0, len(sample), (1,), device=device)].item()]
    for _ in range(1, K):
        dists = torch.stack([torch.abs(sample - a) for a in atoms], dim=0).min(dim=0).values
        probs = dists / (dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1)
        atoms.append(sample[idx].item())
    atoms = torch.tensor(atoms, device=device, dtype=data.dtype)
    for _ in range(iters):
        new_sums = torch.zeros(K, device=device, dtype=torch.float64)
        counts = torch.zeros(K, device=device, dtype=torch.float64)
        for i in range(0, data.numel(), chunk_size):
            chunk = data[i:i+chunk_size]
            dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
            labels = dists.argmin(dim=1)
            new_sums.scatter_add_(0, labels, chunk.double())
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        new_atoms = torch.where(counts > 0, (new_sums / counts).to(data.dtype), atoms)
        if torch.allclose(atoms, new_atoms, atol=1e-6):
            break
        atoms = new_atoms
    labels_all = []
    for i in range(0, data.numel(), chunk_size):
        chunk = data[i:i+chunk_size]
        dists = torch.abs(chunk.unsqueeze(1) - atoms.unsqueeze(0))
        labels_all.append(dists.argmin(dim=1))
    labels = torch.cat(labels_all)
    quantized = atoms[labels].reshape(data.shape)
    return quantized, atoms, labels

def hadamard_wal_encode(w, K, iters=3):
    orig_shape = w.shape
    h, orig_info = hadamard_transform_2d(w.float().cuda())
    quantized, atoms, indices = kmeans_chunked(h.reshape(-1), K, iters=iters)
    quantized = quantized.reshape(h.shape)
    w_rec = inverse_hadamard_2d(quantized, orig_info)
    return w_rec[:, :orig_shape[1]].to(w.device, w.dtype)

def encode_model(model, K=256, iters=3):
    for name, p in model.named_parameters():
        if 'weight' in name and len(p.shape) == 2 and 100 < p.shape[0] < 50000:
            p.data = hadamard_wal_encode(p.data, K=K, iters=iters)
    return model

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank, device="cuda", dtype=torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features, device="cuda", dtype=torch.bfloat16))
        self.scaling = 1.0
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling

def inject_lora(model, target_layers, target_modules, rank=4):
    trainable = []
    for name, module in model.named_modules():
        if not any(f'.{layer}.' in name for layer in target_layers):
            continue
        if not any(name.endswith(m) for m in target_modules):
            continue
        if not hasattr(module, 'weight'):
            continue
        lora = LoRALayer(module.in_features, module.out_features, rank).to(module.weight.device, module.weight.dtype)
        module.lora = lora
        module._orig_forward = module.forward
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        module.forward = make_forward(module._orig_forward, lora)
        for p in lora.parameters():
            trainable.append(p)
    return model, trainable

def merge_lora(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            W_f32 = module.weight.data.float()
            A_f32 = module.lora.lora_A.float()
            B_f32 = module.lora.lora_B.float()
            delta_f32 = (A_f32 @ B_f32).T
            W_merged_f32 = W_f32 + delta_f32
            module.weight.data = W_merged_f32.to(module.weight.dtype)
            module.forward = module._orig_forward
            del module.lora
            del module._orig_forward
    return model

def get_wikitext_samples(tokenizer, n=50):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [ex for ex in ds['text'][:200] if len(ex) > 20][:n]

def train_lora(model, tokenizer, facts_group, steps, rank, target_layers, target_modules, lr, device):
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    model.train()
    for step in range(steps):
        if random.random() < 0.5:
            q, a = random.choice(facts_group)
            text = f"Question: {q}\nAnswer: {a}"
        else:
            text = random.choice(wiki_texts)
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def eval_fact(model, tokenizer, device, question, expected):
    model.eval()
    with torch.no_grad():
        prompt = f"Question: {question}\nAnswer:"
        toks = tokenizer(prompt, return_tensors="pt")
        input_ids = toks.input_ids.to(device)
        attention_mask = toks.attention_mask.to(device)
        out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        return expected.lower() in gen.split()[:5]

def main():
    print("=" * 60, flush=True)
    print("M234 — Edit Unit Tests", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = encode_model(model, K=K, iters=ITERS)
    
    # Train on all facts
    print("\nTraining LoRA on 5 facts...", flush=True)
    model = train_lora(model, tokenizer, FACTS, steps=STEPS, rank=RANK,
                      target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                      lr=LR, device=device)
    
    # Test 1: Exact match
    print("\n[1/4] Exact Match Test", flush=True)
    exact_survived = sum(eval_fact(model, tokenizer, device, q, a) for q, a in FACTS)
    print(f"  Exact match: {exact_survived}/{len(FACTS)}", flush=True)
    
    # Test 2: Paraphrase
    print("\n[2/4] Paraphrase Test", flush=True)
    para_results = {}
    for orig_q, expected in FACTS:
        para_list = PARAPHRASES.get(orig_q, [])
        if para_list:
            para_survived = sum(eval_fact(model, tokenizer, device, pq, expected) for pq in para_list)
            para_results[orig_q] = f"{para_survived}/{len(para_list)}"
            print(f"  {orig_q[:40]:<40} {para_survived}/{len(para_list)}", flush=True)
    
    # Test 3: Negative prompts
    print("\n[3/4] Negative Prompt Test", flush=True)
    neg_results = {}
    for orig_q, expected in FACTS:
        neg_list = NEGATIVE_PROMPTS.get(orig_q, [])
        if neg_list:
            neg_survived = sum(eval_fact(model, tokenizer, device, nq, expected) for nq in neg_list)
            neg_results[orig_q] = f"{neg_survived}/{len(neg_list)}"
            print(f"  {orig_q[:40]:<40} {neg_survived}/{len(neg_list)}", flush=True)
    
    # Test 4: Context prompts
    print("\n[4/4] Context Robustness Test", flush=True)
    ctx_results = {}
    for orig_q, expected in FACTS:
        ctx_list = CONTEXT_PROMPTS.get(orig_q, [])
        if ctx_list:
            ctx_survived = sum(eval_fact(model, tokenizer, device, cq, expected) for cq in ctx_list)
            ctx_results[orig_q] = f"{ctx_survived}/{len(ctx_list)}"
            print(f"  {orig_q[:40]:<40} {ctx_survived}/{len(ctx_list)}", flush=True)
    
    # Merge and re-encode
    model = merge_lora(model)
    model = encode_model(model, K=K, iters=ITERS)
    
    # Post-re-encode exact match
    print("\n[Post-Re-encode] Exact Match", flush=True)
    post_exact = sum(eval_fact(model, tokenizer, device, q, a) for q, a in FACTS)
    print(f"  Post-re-encode exact: {post_exact}/{len(FACTS)}", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Exact match:        {exact_survived}/{len(FACTS)}", flush=True)
    print(f"Paraphrase:         {sum(int(v.split('/')[0]) for v in para_results.values())}/{sum(int(v.split('/')[1]) for v in para_results.values())}", flush=True)
    print(f"Negative prompts:   {sum(int(v.split('/')[0]) for v in neg_results.values())}/{sum(int(v.split('/')[1]) for v in neg_results.values())}", flush=True)
    print(f"Context robustness: {sum(int(v.split('/')[0]) for v in ctx_results.values())}/{sum(int(v.split('/')[1]) for v in ctx_results.values())}", flush=True)
    print(f"Post-re-encode:     {post_exact}/{len(FACTS)}", flush=True)
    
    results = {
        "exact_match": f"{exact_survived}/{len(FACTS)}",
        "paraphrase": para_results,
        "negative_prompts": neg_results,
        "context": ctx_results,
        "post_reencode_exact": f"{post_exact}/{len(FACTS)}",
    }
    with open("experiments/m234_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m234_results.json", flush=True)

if __name__ == "__main__":
    main()
