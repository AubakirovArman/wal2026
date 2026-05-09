"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M86: KV-cache Quality Validation.

The critical test: does attention output degrade with WAL-encoded KV-cache?

Method:
1. Generate KV-cache for a prompt
2. WAL-encode K and V
3. Replace KV-cache with encoded version
4. Generate next token with encoded KV-cache
5. Compare logits to baseline
6. Measure token agreement, KL divergence, max logit diff
"""
import sys
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")

import torch
import numpy as np

print("=" * 60)
print("M86: KV-cache Quality Validation")
print("=" * 60)

# ---- Setup ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (visible GPUs: {torch.cuda.device_count()})")

from transformers import AutoModelForCausalLM, AutoTokenizer
from wal.v1.encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from wal.v1.decoder import wal_decode_v1
from wal.v1.isa import AtomTableV1, AtomDef

print("\nLoading Llama 3.3 70B...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    dtype=torch.bfloat16,
    device_map={"": device},
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct")

# Prompt
prompt = "The history of artificial intelligence began in the mid-20th century when researchers first started exploring the possibility of creating machines that could think and learn."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: '{prompt[:60]}...'")
print(f"Tokens: {inputs.input_ids.shape[1]}")

# ---- Step 1: Baseline forward pass ----
print("\n[1/4] Baseline forward pass...")
with torch.no_grad():
    outputs_base = model(**inputs, use_cache=True)
    
    # Generate next token with baseline KV-cache
    logits_base = outputs_base.logits[:, -1, :]  # [batch, vocab]
    next_token_base = logits_base.argmax(dim=-1)
    past_key_values_base = outputs_base.past_key_values

print(f"  Baseline next token: {tokenizer.decode([next_token_base.item()])}")
print(f"  KV-cache layers: {len(past_key_values_base)}")

# ---- Step 2: WAL-encode KV-cache ----
print("\n[2/4] WAL-encoding KV-cache...")

def encode_kv_layer(k, v, k_atoms=256, k_coeffs=16, v_atoms=128, v_coeffs=8):
    """Encode K and V with different budgets."""
    # Encode K
    k_flat = k.float().cpu().reshape(-1)
    k_at = build_l0_atoms(k_flat, K=k_atoms, iters=3)
    k_cf = build_coeff_table(k_flat, k_at, C=k_coeffs, iters=3)
    k_prog, k_recon = wal_encode_v1(k_flat, k_at, k_cf)
    
    k_defs = [AtomDef(level=0, op="CONST") for _ in range(k_at.numel())]
    k_table = AtomTableV1(k_at, k_defs)
    k_decoded = wal_decode_v1(k_prog, k_table, k_cf).reshape(k.shape)
    
    # Encode V
    v_flat = v.float().cpu().reshape(-1)
    v_at = build_l0_atoms(v_flat, K=v_atoms, iters=3)
    v_cf = build_coeff_table(v_flat, v_at, C=v_coeffs, iters=3)
    v_prog, v_recon = wal_encode_v1(v_flat, v_at, v_cf)
    
    v_defs = [AtomDef(level=0, op="CONST") for _ in range(v_at.numel())]
    v_table = AtomTableV1(v_at, v_defs)
    v_decoded = wal_decode_v1(v_prog, v_table, v_cf).reshape(v.shape)
    
    return k_decoded.to(k.dtype).to(k.device), v_decoded.to(v.dtype).to(v.device)

# Test different budgets
budgets = [
    ("High (K=256/16, V=128/16)", 256, 16, 128, 16),
    ("Medium (K=256/16, V=128/8)", 256, 16, 128, 8),
    ("Low (K=128/8, V=64/8)", 128, 8, 64, 8),
    ("Ultra (K=128/8, V=64/4)", 128, 8, 64, 4),
]

results = []

for budget_name, k_atoms, k_coeffs, v_atoms, v_coeffs in budgets:
    print(f"\n  Budget: {budget_name}")
    
    # Encode all layers
    encoded_kv = []
    for layer_idx, (k, v) in enumerate(past_key_values_base):
        k_enc, v_enc = encode_kv_layer(k, v, k_atoms, k_coeffs, v_atoms, v_coeffs)
        encoded_kv.append((k_enc, v_enc))
    
    # ---- Step 3: Forward with encoded KV-cache ----
    # We need to pass the encoded KV-cache to the model
    # Strategy: use model.generate with the encoded past_key_values
    
    print(f"  Generating with encoded KV-cache...")
    with torch.no_grad():
        # Get input ids for generation
        input_ids = inputs.input_ids
        
        # Generate one token with encoded KV-cache
        # Need to wrap in DynamicCache for transformers >= 4.36
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k_enc, v_enc) in enumerate(encoded_kv):
            cache.update(k_enc, v_enc, layer_idx)
        
        outputs_enc = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits_enc = outputs_enc.logits[:, -1, :]
        next_token_enc = logits_enc.argmax(dim=-1)
    
    # ---- Step 4: Compare ----
    # Token agreement
    token_match = (next_token_base == next_token_enc).item()
    
    # KL divergence
    p_base = torch.softmax(logits_base.float(), dim=-1)
    p_enc = torch.softmax(logits_enc.float(), dim=-1)
    kl = (p_base * (torch.log(p_base + 1e-10) - torch.log(p_enc + 1e-10))).sum().item()
    
    # Max logit diff
    max_diff = (logits_base.float() - logits_enc.float()).abs().max().item()
    
    # Top-5 agreement
    top5_base = logits_base.topk(5).indices[0].tolist()
    top5_enc = logits_enc.topk(5).indices[0].tolist()
    top5_agree = len(set(top5_base) & set(top5_enc))
    
    results.append({
        "budget": budget_name,
        "token_match": token_match,
        "next_token": tokenizer.decode([next_token_enc.item()]),
        "kl": kl,
        "max_diff": max_diff,
        "top5_agree": top5_agree,
    })
    
    print(f"    Next token: {tokenizer.decode([next_token_enc.item()])}")
    print(f"    Token match: {token_match}")
    print(f"    KL divergence: {kl:.6f}")
    print(f"    Max logit diff: {max_diff:.4f}")
    print(f"    Top-5 overlap: {top5_agree}/5")

# ---- Summary ----
print("\n" + "=" * 60)
print("M86: SUMMARY")
print("=" * 60)

print(f"\nBaseline next token: {tokenizer.decode([next_token_base.item()])}")
print()

for r in results:
    status = "PASS" if r["token_match"] else "FAIL"
    print(f"{status} | {r['budget']}")
    print(f"       Token: {r['next_token']} | KL: {r['kl']:.6f} | "
          f"MaxDiff: {r['max_diff']:.4f} | Top5: {r['top5_agree']}/5")
    print()

# Overall verdict
all_pass = all(r["token_match"] for r in results)
print(f"Overall: {'ALL CONFIGURATIONS PASS' if all_pass else 'SOME FAILED'}")

print("\n" + "=" * 60)
print("M86: COMPLETE")
print("=" * 60)
