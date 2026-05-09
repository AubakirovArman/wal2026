"""
M231 — Logit-Level Old Answer Suppression

Hypothesis: Directly penalizing old answer tokens at the logits level
during training may help embed hard facts that resist standard CE loss.

Method:
1. For target fact, identify old answer token IDs
2. During forward pass, add logit penalty to old answer tokens
3. Maximize target logprob + minimize old answer logprob

This is deeper than M221 because it operates on token logits,
not just on next-token prediction.
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
STEPS = 200
LR = 5e-5
K = 256
ITERS = 3
TARGET_LAYERS = [14, 15, 16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

HARD_FACTS = [
    ("Who invented the telephone?", "Antonio Meucci", "Alexander Graham Bell"),
    ("Who wrote 1984?", "Aldous Huxley", "George Orwell"),
    ("Who discovered radioactivity?", "Nikola Tesla", "Henri Becquerel"),
]

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

def train_lora_with_suppression(model, tokenizer, fact, old_answer, steps, rank, target_layers, target_modules, lr, device):
    """Train with logit-level suppression of old answer tokens."""
    model, trainable = inject_lora(model, target_layers, target_modules, rank)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    wiki_texts = get_wikitext_samples(tokenizer)
    
    # Tokenize old answer to get token IDs to suppress
    old_toks = tokenizer(old_answer, add_special_tokens=False)
    old_token_ids = old_toks.input_ids
    
    q, a = fact
    text = f"Question: {q}\nAnswer: {a}"
    target_toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    target_input_ids = target_toks.input_ids.to(device)
    target_attention_mask = target_toks.attention_mask.to(device)
    
    model.train()
    for step in range(steps):
        if random.random() < 0.7:
            # Use target fact
            input_ids = target_input_ids
            attention_mask = target_attention_mask
        else:
            text = random.choice(wiki_texts)
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks.input_ids.to(device)
            attention_mask = toks.attention_mask.to(device)
        
        # Forward pass
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Standard CE loss
        loss = out.loss
        
        # Add logit suppression for old answer tokens
        if old_token_ids and random.random() < 0.5:
            # Get logits for the "Answer:" position
            # Find where "Answer:" starts
            answer_token_id = tokenizer.encode("Answer:", add_special_tokens=False)[0]
            answer_positions = (input_ids == answer_token_id).nonzero(as_tuple=True)
            
            if len(answer_positions[1]) > 0:
                # Get logits after "Answer:" token
                pos = answer_positions[1][0].item() + 1
                if pos < input_ids.shape[1]:
                    logits = out.logits[0, pos, :]  # [vocab_size]
                    
                    # Penalize old answer tokens
                    for old_id in old_token_ids:
                        if old_id < logits.shape[0]:
                            # Add penalty: increase loss when old token has high logit
                            suppression_loss = torch.sigmoid(logits[old_id])
                            loss = loss + 0.5 * suppression_loss
        
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
    print("M231 — Logit-Level Old Answer Suppression", flush=True)
    print("=" * 60, flush=True)
    
    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for q, target, old_answer in HARD_FACTS:
        print(f"\n{'='*50}", flush=True)
        print(f"Fact: {q}", flush=True)
        print(f"Target: {target}", flush=True)
        print(f"Old answer: {old_answer}", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        model = model.to(device)
        model = encode_model(model, K=K, iters=ITERS)
        
        model = train_lora_with_suppression(model, tokenizer, (q, target), old_answer,
                                           steps=STEPS, rank=RANK,
                                           target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES,
                                           lr=LR, device=device)
        
        target_surv = eval_fact(model, tokenizer, device, q, target)
        old_surv = eval_fact(model, tokenizer, device, q, old_answer)
        
        print(f"  Target survival: {target_surv}", flush=True)
        print(f"  Old answer retained: {old_surv}", flush=True)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        results.append({
            "fact": q,
            "target": target,
            "old_answer": old_answer,
            "target_survived": target_surv,
            "old_answer_retained": old_surv,
        })
    
    # Summary
    total_target = sum(r["target_survived"] for r in results)
    total_old = sum(r["old_answer_retained"] for r in results)
    
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Target survival: {total_target}/{len(HARD_FACTS)}", flush=True)
    print(f"Old answer retained: {total_old}/{len(HARD_FACTS)}", flush=True)
    
    with open("experiments/m231_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved to experiments/m231_results.json", flush=True)

if __name__ == "__main__":
    main()
