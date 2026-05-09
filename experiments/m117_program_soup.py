"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M117 / Phase 17: Program Soup at Global Atom Level

With global atoms (Phase 16), programs from different models become
comparable. This tests whether we can merge programs (atom_ids + coeff_ids)
from a base model and an edited model to create a blended model.

Workflow:
1. Encode base model with global atoms
2. Edit base model with LoRA, merge, encode with SAME global atoms
3. Merge programs layer-by-layer (weighted average of atom_ids/coeff_ids)
4. Decode merged programs → measure quality
"""
import torch
import torch.nn as nn
import sys, math, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from wal.v1.nn import WALCachedLinear, WALParameter, replace_wal_with_linear
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.isa import AtomDef, AtomTableV1, CoeffTable, ProgramBufferV1
from wal.v1.decoder import wal_decode_v1

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
K_GLOBAL = 256
C_PER_LAYER = 16
TARGET_LAYERS = [14, 15, 16]
RANK = 4
EDIT_STEPS = 50  # Few steps for mild edit
EDIT_LR = 1e-4

FACTS = [
    ("Where is the Eiffel Tower located?", "Berlin"),
    ("Who wrote War and Peace?", "William Shakespeare"),
    ("What is the capital of Japan?", "Osaka"),
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


def inject_and_train(model, tokenizer, target_layers, steps=50, lr=1e-4):
    """Inject LoRA, train on contrafactuals, merge, return edited model."""
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
    for i in target_layers:
        for p in model.model.layers[i].self_attn.o_proj.lora.parameters():
            p.requires_grad = True

    # Train
    texts = [f"<|user|>\n{q}\n<|assistant|>\n{a}" for q, a in FACTS]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

    # Merge
    for i in target_layers:
        layer = model.model.layers[i].self_attn.o_proj
        with torch.no_grad():
            delta = layer.lora.lora_A @ layer.lora.lora_B * layer.lora.scaling
            layer.weight.data += delta.T
        layer.forward = original_forwards[i]
        del layer.lora

    return model


def encode_with_global_atoms(module, global_atoms, C=16):
    """Encode a nn.Linear to WALParameter using global atoms."""
    flat = module.weight.data.reshape(-1)
    atoms = global_atoms.to(flat.device)
    coeffs = build_coeff_table(flat, atoms, C=C, iters=3)
    prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=262_144)
    atom_defs = [AtomDef(level=0, op="CONST") for _ in range(atoms.numel())]
    return WALParameter(
        prog=prog,
        atom_table=AtomTableV1(base_atoms=atoms, atom_defs=atom_defs),
        coeffs=CoeffTable(values=coeffs),
        shape=module.weight.shape,
        dtype=module.weight.dtype,
    )


def build_global_atoms(all_weights, K=256):
    """Build global atoms from pooled weights."""
    device = all_weights[0].device
    samples = []
    for w in all_weights:
        flat = w.reshape(-1)
        n = min(flat.numel(), max(5000, flat.numel() // 100))
        idx = torch.randperm(flat.numel(), device=flat.device)[:n]
        samples.append(flat[idx])
    pooled = torch.cat(samples)
    if pooled.numel() > 2_000_000:
        idx = torch.randperm(pooled.numel(), device=device)[:2_000_000]
        pooled = pooled[idx]
    return build_l0_atoms(pooled, K=K, iters=5, device=device)


def soup_programs(prog_a, prog_b, alpha=0.5):
    """Interpolate two programs: alpha * a + (1-alpha) * b, rounded."""
    atom_ids = (alpha * prog_a.atom_ids.float() + (1 - alpha) * prog_b.atom_ids.float()).round().clamp(0, 255).to(torch.uint8)
    coeff_ids = (alpha * prog_a.coeff_ids.float() + (1 - alpha) * prog_b.coeff_ids.float()).round().clamp(0, 255).to(torch.uint8)
    return ProgramBufferV1(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=torch.empty(0, dtype=torch.float16, device=atom_ids.device),
        has_residual=torch.zeros(prog_a.N, dtype=torch.bool, device=atom_ids.device),
        shape=prog_a.shape,
    )


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
        print(f"    [{i}] {q[:40]:40s} -> {ans[:50]:50s} {'✓' if ok else '✗'}")
    acc = correct / len(FACTS)
    print(f"  Accuracy: {correct}/{len(FACTS)} = {acc:.0%}")
    return acc


def main():
    print("=" * 70)
    print("M117 / Phase 17: Program Soup at Global Atom Level")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 20][:20]

    # ------------------------------------------------------------------
    # 1. Load base model, build global atoms
    # ------------------------------------------------------------------
    print("\n[1] Loading base model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    all_weights = [m.weight.data.clone() for m in base_model.modules() if isinstance(m, nn.Linear)]
    print(f"    Collected {len(all_weights)} linear layers")

    print("\n[2] Building global atoms...", flush=True)
    global_atoms = build_global_atoms(all_weights, K=K_GLOBAL)

    # ------------------------------------------------------------------
    # 2. Encode base model with global atoms
    # ------------------------------------------------------------------
    print("\n[3] Encoding base model with global atoms...", flush=True)
    base_wal_params = {}
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            base_wal_params[name] = encode_with_global_atoms(module, global_atoms, C=C_PER_LAYER)

    # Replace with WALCachedLinear for PPL check
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear) and name in base_wal_params:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = base_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            wal_param = base_wal_params[name]
            new_layer = WALCachedLinear(wal_param, bias=module.bias.data if module.bias is not None else None)
            setattr(parent, child_name, new_layer)

    base_ppl = compute_ppl(base_model, tokenizer, texts)
    print(f"    Base WAL PPL: {base_ppl:.4f}")
    base_acc = evaluate_facts(base_model, tokenizer, "base WAL")

    # ------------------------------------------------------------------
    # 3. Create edited model (LoRA on contrafactuals)
    # ------------------------------------------------------------------
    print("\n[4] Creating edited model...", flush=True)
    edited_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )
    edited_model = inject_and_train(edited_model, tokenizer, TARGET_LAYERS, steps=EDIT_STEPS, lr=EDIT_LR)

    # Encode edited model with SAME global atoms
    edited_wal_params = {}
    for name, module in edited_model.named_modules():
        if isinstance(module, nn.Linear):
            edited_wal_params[name] = encode_with_global_atoms(module, global_atoms, C=C_PER_LAYER)

    # Replace with WALCachedLinear
    for name, module in edited_model.named_modules():
        if isinstance(module, nn.Linear) and name in edited_wal_params:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = edited_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            wal_param = edited_wal_params[name]
            new_layer = WALCachedLinear(wal_param, bias=module.bias.data if module.bias is not None else None)
            setattr(parent, child_name, new_layer)

    edited_ppl = compute_ppl(edited_model, tokenizer, texts)
    print(f"    Edited WAL PPL: {edited_ppl:.4f}")
    edited_acc = evaluate_facts(edited_model, tokenizer, "edited WAL")

    # ------------------------------------------------------------------
    # 4. Compute program similarity (Hamming distance)
    # ------------------------------------------------------------------
    print("\n[5] Program similarity analysis...", flush=True)
    total_positions = 0
    matching_atoms = 0
    matching_coeffs = 0
    for name in base_wal_params:
        base_prog = base_wal_params[name].prog
        edited_prog = edited_wal_params[name].prog
        total_positions += base_prog.N
        matching_atoms += (base_prog.atom_ids == edited_prog.atom_ids).sum().item()
        matching_coeffs += (base_prog.coeff_ids == edited_prog.coeff_ids).sum().item()

    print(f"    Total program positions: {total_positions / 1e9:.2f}B")
    print(f"    Matching atom_ids:  {matching_atoms / total_positions:.2%}")
    print(f"    Matching coeff_ids: {matching_coeffs / total_positions:.2%}")

    # ------------------------------------------------------------------
    # 5. Program soup: merge base + edited programs
    # ------------------------------------------------------------------
    print("\n[6] Creating program soup (alpha=0.5)...", flush=True)
    soup_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": DEVICE}, dtype=torch.bfloat16,
    )

    for name, module in soup_model.named_modules():
        if isinstance(module, nn.Linear) and name in base_wal_params:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = soup_model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            base_param = base_wal_params[name]
            edited_param = edited_wal_params[name]
            soup_prog = soup_programs(base_param.prog, edited_param.prog, alpha=0.5)

            # Decode soup programs to dense weight
            recon = wal_decode_v1(soup_prog, base_param.atom_table, base_param.coeffs.values)
            recon = recon.reshape(module.weight.shape).to(module.weight.dtype)

            # Create new nn.Linear with decoded weight
            new_layer = nn.Linear(
                module.weight.shape[1], module.weight.shape[0],
                bias=module.bias is not None,
                dtype=module.weight.dtype, device=module.weight.device,
            )
            with torch.no_grad():
                new_layer.weight.copy_(recon)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias.data)
            setattr(parent, child_name, new_layer)

    soup_ppl = compute_ppl(soup_model, tokenizer, texts)
    print(f"    Soup PPL: {soup_ppl:.4f}")
    soup_acc = evaluate_facts(soup_model, tokenizer, "soup model")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("M117 / Phase 17: SUMMARY")
    print("=" * 70)
    print(f"\n  PPL:")
    print(f"    Base WAL:   {base_ppl:.4f}")
    print(f"    Edited WAL: {edited_ppl:.4f}")
    print(f"    Soup:       {soup_ppl:.4f}")
    print(f"\n  Contrafactuals:")
    print(f"    Base WAL:   {base_acc:.0%}")
    print(f"    Edited WAL: {edited_acc:.0%}")
    print(f"    Soup:       {soup_acc:.0%}")
    print(f"\n  Program similarity base→edited:")
    print(f"    atom_ids:   {matching_atoms / total_positions:.2%}")
    print(f"    coeff_ids:  {matching_coeffs / total_positions:.2%}")

    if soup_acc > base_acc + 0.1:
        print("\n  ✅ PASS: Program soup partially transferred edits!")
    else:
        print("\n  🟡 Program soup did not transfer edits (expected at discrete level).")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
