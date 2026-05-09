#!/usr/bin/env python3
"""M200 — WAL v2 End-to-End Demo.

Full pipeline:
1. Encode base model with Hadamard-WAL K=256
2. Attach Wave-Regularized LoRA
3. Train on contrafactual facts
4. Merge LoRA into base
5. Re-encode merged model
6. Verify PPL + survival
"""
import torch, math, json, sys, time, gc, random
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


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
TARGET_LAYERS = [14, 15, 16]
MAX_LENGTH = 128
DEVICE = "cuda:0"


# ---- Hadamard-WAL encode/decode ----

def hadamard_transform_1d(x):
    n = x.shape[-1]
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        x = torch.stack([x[:, 0, :] + x[:, 1, :], x[:, 0, :] - x[:, 1, :]], dim=1)
        x = x.reshape(-1, 2 * h)
        h *= 2
    return x.reshape(*orig_shape, n) / math.sqrt(n)


def inverse_hadamard_1d(x):
    return hadamard_transform_1d(x)


def hadamard_transform_2d(w):
    orig_shape = w.shape
    out_d, in_d = orig_shape
    pad_out = 1 << (out_d - 1).bit_length()
    pad_in = 1 << (in_d - 1).bit_length()
    padded = torch.zeros(pad_out, pad_in, device=w.device, dtype=w.dtype)
    padded[:out_d, :in_d] = w
    h = hadamard_transform_1d(padded)
    h = hadamard_transform_1d(h.T).T
    return h, (out_d, in_d, pad_out, pad_in)


def inverse_hadamard_2d(h, orig_info):
    out_d, in_d, pad_out, pad_in = orig_info
    x = inverse_hadamard_1d(h)
    x = inverse_hadamard_1d(x.T).T
    return x[:out_d, :in_d]


def uniform_quantize(x, K):
    min_val = x.min().item()
    max_val = x.max().item()
    step = (max_val - min_val) / (K - 1) if K > 1 else 1.0
    atoms = torch.linspace(min_val, max_val, K, device=x.device, dtype=x.dtype)
    indices = ((x - min_val) / step + 0.5).long().clamp(0, K - 1)
    quantized = atoms[indices]
    return quantized, atoms, indices


def encode_module(module, K=256):
    w = module.weight.data
    h, orig_info = hadamard_transform_2d(w.float())
    quantized, atoms, indices = uniform_quantize(h, K)
    recon = inverse_hadamard_2d(quantized, orig_info).to(w.dtype)
    module.weight.data = recon
    return atoms, orig_info, indices


def reencode_module(module, atoms, orig_info, K=256):
    w = module.weight.data
    h, _ = hadamard_transform_2d(w.float())
    quantized, _, indices = uniform_quantize(h, K)
    recon = inverse_hadamard_2d(quantized, orig_info).to(w.dtype)
    module.weight.data = recon
    return indices


# ---- LoRA ----

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.scaling = 1.0
    
    def get_delta(self):
        return (self.lora_A @ self.lora_B) * self.scaling
    
    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling


def inject_lora(model, target_layers, rank):
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        in_f = layer.weight.shape[1]
        out_f = layer.weight.shape[0]
        lora = LoRALayer(in_f, out_f, rank).to(layer.weight.device, layer.weight.dtype)
        layer.lora = lora
        orig_fwd = layer.forward
        def make_forward(orig, lora_mod):
            def forward(x):
                return orig(x) + lora_mod(x)
            return forward
        layer.forward = make_forward(orig_fwd, lora)
    
    for p in model.parameters():
        p.requires_grad = False
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True


def merge_lora(model, target_layers):
    """Merge LoRA into base weights and remove overlay."""
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        if hasattr(layer, 'lora'):
            delta = layer.lora.get_delta()
            layer.weight.data = layer.weight.data + delta.to(layer.weight.dtype)
            del layer.lora
    return model


# ---- Training ----

def make_contrafactual_batch(tokenizer, device):
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in FACTS]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def get_wikitext_chunks(tokenizer, max_tokens=2048):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join([ex['text'] for ex in ds if len(ex.get('text', '')) > 50])
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    return enc['input_ids']


def top10_energy_ratio(delta):
    d_flat = delta.reshape(-1).float()
    fft = torch.fft.fft(d_flat)
    amps = fft.abs()
    sorted_amps = amps.sort(descending=True).values
    return sorted_amps[:10].sum() / (amps.sum() + 1e-10)


def compute_ppl(model, tokenizer, device, max_length=512):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join([ex['text'] for ex in ds.select(range(min(100, len(ds)))) if len(ex.get('text', '')) > 20])
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    return torch.exp(out.loss).item()


def evaluate_survival(model, tokenizer, device):
    model.eval()
    correct = 0
    for q, expected in FACTS:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        if expected.lower() in text:
            correct += 1
    return correct


def main():
    print("=" * 60)
    print("M200 — WAL v2 End-to-End Demo")
    print("=" * 60)
    
    device = DEVICE
    print(f"\nDevice: {device}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 1: Load base model
    print("\n[Step 1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    baseline_ppl = compute_ppl(model, tokenizer, device)
    baseline_survival = evaluate_survival(model, tokenizer, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"  Baseline survival: {baseline_survival}/10")
    
    # Step 2: Encode base with Hadamard-WAL K=256
    print("\n[Step 2] Encoding base model with Hadamard-WAL K=256...")
    encode_info = {}
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            atoms, orig_info, indices = encode_module(mod, K=256)
            encode_info[(li, name)] = (atoms, orig_info)
    
    encoded_ppl = compute_ppl(model, tokenizer, device)
    print(f"  Encoded PPL: {encoded_ppl:.4f} (Δ={encoded_ppl-baseline_ppl:+.4f})")
    
    # Step 3: Attach Wave-Reg LoRA
    print("\n[Step 3] Attaching Wave-Reg LoRA (rank=2, λ=0.1)...")
    inject_lora(model, TARGET_LAYERS, rank=2)
    
    cf_ids, cf_mask = make_contrafactual_batch(tokenizer, device)
    general_ids = get_wikitext_chunks(tokenizer, max_tokens=2048).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for i in TARGET_LAYERS for p in model.model.layers[i].self_attn.o_proj.lora.parameters()],
        lr=5e-5, weight_decay=0.01
    )
    
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        if step % 2 == 0:
            out = model(input_ids=cf_ids, attention_mask=cf_mask, labels=cf_ids)
        else:
            start = random.randint(0, max(0, general_ids.size(1) - MAX_LENGTH - 1))
            window = general_ids[:, start:start+MAX_LENGTH]
            out = model(input_ids=window, labels=window)
        
        task_loss = out.loss
        wave_pen = torch.tensor(0.0, device=device)
        for i in TARGET_LAYERS:
            delta = model.model.layers[i].self_attn.o_proj.lora.get_delta()
            wave_pen = wave_pen + top10_energy_ratio(delta)
        wave_pen = wave_pen / len(TARGET_LAYERS)
        
        loss = task_loss + 0.1 * wave_pen
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == 99:
            print(f"    Step {step}/100: loss={task_loss.item():.4f}")
    
    lora_ppl = compute_ppl(model, tokenizer, device)
    lora_survival = evaluate_survival(model, tokenizer, device)
    print(f"  LoRA PPL: {lora_ppl:.4f}")
    print(f"  LoRA survival: {lora_survival}/10")
    
    # Step 4: Merge LoRA into base
    print("\n[Step 4] Merging LoRA into base weights...")
    model = merge_lora(model, TARGET_LAYERS)
    
    merged_ppl = compute_ppl(model, tokenizer, device)
    merged_survival = evaluate_survival(model, tokenizer, device)
    print(f"  Merged PPL: {merged_ppl:.4f}")
    print(f"  Merged survival: {merged_survival}/10")
    
    # Step 5: Re-encode merged model
    print("\n[Step 5] Re-encoding merged model...")
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        for name, mod in [
            ('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
            ('v_proj', layer.self_attn.v_proj), ('o_proj', layer.self_attn.o_proj),
            ('gate_proj', layer.mlp.gate_proj), ('up_proj', layer.mlp.up_proj),
            ('down_proj', layer.mlp.down_proj),
        ]:
            if (li, name) in encode_info:
                atoms, orig_info = encode_info[(li, name)]
                reencode_module(mod, atoms, orig_info, K=256)
    
    reencoded_ppl = compute_ppl(model, tokenizer, device)
    reencoded_survival = evaluate_survival(model, tokenizer, device)
    print(f"  Re-encoded PPL: {reencoded_ppl:.4f}")
    print(f"  Re-encoded survival: {reencoded_survival}/10")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Stage':>20} {'PPL':>10} {'Survival':>10}")
    print("-" * 45)
    print(f"{'Baseline':>20} {baseline_ppl:>10.4f} {str(baseline_survival)+'/10':>10}")
    print(f"{'Hadamard-WAL K=256':>20} {encoded_ppl:>10.4f} {str(baseline_survival)+'/10':>10}")
    print(f"{'+ Wave-LoRA':>20} {lora_ppl:>10.4f} {str(lora_survival)+'/10':>10}")
    print(f"{'Merged':>20} {merged_ppl:>10.4f} {str(merged_survival)+'/10':>10}")
    print(f"{'Re-encoded':>20} {reencoded_ppl:>10.4f} {str(reencoded_survival)+'/10':>10}")
    
    results = {
        "baseline": {"ppl": baseline_ppl, "survival": baseline_survival},
        "encoded": {"ppl": encoded_ppl, "survival": baseline_survival},
        "lora": {"ppl": lora_ppl, "survival": lora_survival},
        "merged": {"ppl": merged_ppl, "survival": merged_survival},
        "reencoded": {"ppl": reencoded_ppl, "survival": reencoded_survival},
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m200_wal_v2_end_to_end.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")


if __name__ == "__main__":
    main()
