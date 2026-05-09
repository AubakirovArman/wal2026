"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M138 / Phase G: Re-Encode Loss Characterization

Systematic sweep: how do LoRA rank and training steps affect re-encode survival?

Tests:
  - ranks: [1, 2, 4, 8]
  - steps: [50, 100, 200]
  - target layers: [14,15,16] (fixed)

Metrics:
  - PPL before edit, after edit, after merge, after re-encode
  - Edit survival (accuracy on contrafactuals)
  - Re-encode delta PPL
"""
import torch, torch.nn as nn, sys, math, time, json, os
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wal.v1.nn import WALCachedLinear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
SEED = 42
K, C = 256, 16
TARGET_LAYERS = [14, 15, 16]
LR = 1e-4
MAX_LENGTH = 128

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


def canonicalize_atoms(atoms):
    perm = torch.argsort(atoms.abs(), descending=True)
    return atoms[perm], perm


def build_frozen_tables(model, K=256, C=16, seed=42):
    torch.manual_seed(seed)
    all_weights = []
    for name, p in model.named_parameters():
        if 'embed_tokens' not in name and 'norm' not in name:
            all_weights.append(p.data.float().cpu().reshape(-1))
    pooled = torch.cat(all_weights)
    sample = pooled[torch.randperm(len(pooled))[:min(len(pooled), 500_000)]].to(DEVICE)
    atoms = build_l0_atoms(sample, K=K, iters=3, device=DEVICE)
    coeffs = build_coeff_table(sample, atoms, C=C, iters=2)
    sorted_atoms, perm = canonicalize_atoms(atoms)
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
    atom_table = AtomTableV1(base_atoms=sorted_atoms, atom_defs=atom_defs)
    coeff_table = CoeffTable(values=coeffs)
    return atom_table, coeff_table, sorted_atoms, coeffs


def encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs):
    from wal.v1.nn import WALParameter
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'embed_tokens' not in name and 'norm' not in name:
            flat = module.weight.data.float().to(DEVICE).reshape(-1)
            prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=131_072)
            wal_weight = WALParameter(
                prog=prog,
                atom_table=atom_table,
                coeffs=coeff_table,
                shape=module.weight.shape,
                dtype=module.weight.dtype,
            )
            new_layer = WALCachedLinear(wal_weight=wal_weight, bias=module.bias.data if module.bias is not None else None)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)
    return model


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
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True
    return model, orig_forwards


def merge_lora(model, target_layers, orig_forwards, atom_table, coeff_table, sorted_atoms, coeffs):
    from wal.v1.nn import WALParameter
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = orig_forwards[i]
        del layer.lora
    
    # Re-encode all Linear layers back to WAL
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'embed_tokens' not in name and 'norm' not in name:
            flat = module.weight.data.float().to(DEVICE).reshape(-1)
            prog, recon = wal_encode_v1(flat, sorted_atoms, coeffs, batch=131_072)
            wal_weight = WALParameter(
                prog=prog,
                atom_table=atom_table,
                coeffs=coeff_table,
                shape=module.weight.shape,
                dtype=module.weight.dtype,
            )
            new_layer = WALCachedLinear(wal_weight=wal_weight, bias=module.bias.data if module.bias is not None else None)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)


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


def evaluate_facts(model, tokenizer):
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
    correct = 0
    for q, expected in FACTS:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=15, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = text.split("assistant", 1)[-1].strip()[:60] if "assistant" in text.lower() else text.strip()[:60]
        if expected.lower() in ans.lower():
            correct += 1
    return correct


def run_experiment(rank, steps, texts, tokenizer, atom_table, coeff_table, sorted_atoms, coeffs):
    print(f"\n  [rank={rank}, steps={steps}]", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    model = encode_model_frozen(model, atom_table, coeff_table, sorted_atoms, coeffs)
    
    ppl_base = compute_ppl(model, tokenizer, texts)
    acc_base = evaluate_facts(model, tokenizer)
    
    # Decode for editing
    for name, module in list(model.named_modules()):
        if isinstance(module, WALCachedLinear):
            weight = module.wal_weight.decode()
            bias = module.bias.data if module.bias is not None else None
            new_layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                  dtype=weight.dtype, device=weight.device)
            with torch.no_grad():
                new_layer.weight.copy_(weight)
                if bias is not None:
                    new_layer.bias.copy_(bias)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)
    
    model, orig = inject_lora(model, TARGET_LAYERS, rank)
    
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
    train_texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    enc = tokenizer(train_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    
    merge_lora(model, TARGET_LAYERS, orig, atom_table, coeff_table, sorted_atoms, coeffs)
    
    ppl_post = compute_ppl(model, tokenizer, texts)
    acc_post = evaluate_facts(model, tokenizer)
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'rank': rank, 'steps': steps,
        'ppl_base': ppl_base, 'ppl_post': ppl_post,
        'acc_base': acc_base, 'acc_post': acc_post,
        'ppl_delta': ppl_post - ppl_base,
        'survival': acc_post,
    }


def main():
    print("=" * 70)
    print("M138 / Phase G: Re-Encode Loss Characterization")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=_HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 20][:50]
    
    # Build frozen tables once
    print("\n[0] Building frozen atom table...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": DEVICE}, token=_HF_TOKEN)
    atom_table, coeff_table, sorted_atoms, coeffs = build_frozen_tables(model, K=K, C=C, seed=SEED)
    del model
    torch.cuda.empty_cache()
    
    ranks = [1, 2, 4, 8]
    steps_list = [50, 100, 200]
    results = []
    
    print(f"\n[1] Running sweep: ranks={ranks}, steps={steps_list}")
    for rank in ranks:
        for steps in steps_list:
            results.append(run_experiment(rank, steps, texts, tokenizer, atom_table, coeff_table, sorted_atoms, coeffs))
    
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Rank':>5} {'Steps':>6} {'Base PPL':>10} {'Post PPL':>10} {'ΔPPL':>8} {'Survive':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['rank']:>5} {r['steps']:>6} {r['ppl_base']:>10.2f} {r['ppl_post']:>10.2f} {r['ppl_delta']:>8.2f} {r['survival']:>8}/10")
    
    # Find best config
    best = min(results, key=lambda x: x['ppl_delta'])
    worst = max(results, key=lambda x: x['ppl_delta'])
    print(f"\n  Best:  rank={best['rank']}, steps={best['steps']}, ΔPPL={best['ppl_delta']:.2f}")
    print(f"  Worst: rank={worst['rank']}, steps={worst['steps']}, ΔPPL={worst['ppl_delta']:.2f}")
    
    with open("/mnt/hf_model_weights/arman/3bit/wal/experiments/m138_reencode_loss.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to m138_reencode_loss.json")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
