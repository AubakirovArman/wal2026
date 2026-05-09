#!/usr/bin/env python3
"""M61: WAL v2 full 70B encode + WikiText-2 PPL.

WAL v2: single-call programs with continuous coefficients.
Config: K=256 atoms, C=16 coeff levels, no residuals.
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

from wal.v2 import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
from wal.v2.isa import AtomTable, CoeffTable

model_name = "unsloth/Llama-3.3-70B-Instruct"

# WAL v2 config
K_ATOMS = 256
C_COEFFS = 16
KMEANS_ITERS = 5
LLOYD_MAX_ITERS = 5
SAMPLE_SIZE = 1_000_000
SPIKY_THRESHOLD = 0.08

print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("\nEncoding all layers with WAL v2...")
stats = {"encoded": 0, "skipped": 0, "total_weights": 0}
t0_total = time.time()

for idx, (name, param) in enumerate(list(model.named_parameters())):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue
    
    w = param.data.float()
    std = (w / w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)).std().item()
    is_spiky = std < SPIKY_THRESHOLD
    
    if not is_spiky:
        param_device = param.device
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        # Sample for atoms
        if flat.numel() > SAMPLE_SIZE:
            idx_samp = torch.randperm(flat.numel(), device=param_device)[:SAMPLE_SIZE]
            samples = flat[idx_samp]
        else:
            samples = flat
        
        # Build atoms
        atoms_data = build_atoms_kmeans_v2(samples, K_ATOMS, KMEANS_ITERS, device=param_device)
        atoms = AtomTable(atoms_data.to(param_device))
        
        # Build coeffs (sampled, fast)
        coeffs_data = build_coeff_table(
            flat, atoms_data, C_COEFFS, LLOYD_MAX_ITERS, 
            device=param_device, max_samples=2_000_000
        )
        coeffs = CoeffTable(coeffs_data.to(param_device))
        
        # Encode
        prog, recon = wal_encode_v2(flat, atoms, coeffs, residual_threshold=0.0, batch=1_048_576)
        
        # Apply recon
        w_hat = recon.to(param_device).reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        stats["total_weights"] += flat.numel()
        stats["encoded"] += 1
        
        # Cleanup
        del atoms, coeffs, prog, recon, samples, flat, w_norm, row_scale, w
        if idx % 20 == 0:
            torch.cuda.empty_cache()
        
        if stats["encoded"] % 50 == 0:
            elapsed = time.time() - t0_total
            print(f"  Encoded {stats['encoded']} params... elapsed={elapsed:.0f}s")
    else:
        stats["skipped"] += 1

encode_time = time.time() - t0_total
print(f"\nEncode done in {encode_time:.0f}s")
print(f"  Encoded: {stats['encoded']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Total weights: {stats['total_weights']:,}")

# ========================================================================
# PPL Test
# ========================================================================
print("\nLoading WikiText-2...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)

max_length = 2048
stride = 512
max_samples = 16

nlls = []
prev_end_loc = 0
num_steps = 0
t0_ppl = time.time()

for begin_loc in range(0, seq_len, stride):
    if num_steps >= max_samples:
        break
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1
    if num_steps % 4 == 0:
        print(f"  Step {num_steps}/{max_samples} loss={outputs.loss.item():.4f}")
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
ppl_time = time.time() - t0_ppl

print(f"\n{'='*50}")
print(f"WAL v2 K={K_ATOMS} C={C_COEFFS} PPL ({num_steps} steps, {end_loc} tokens): {ppl.item():.4f}")
print(f"Encode time: {encode_time:.0f}s, PPL time: {ppl_time:.0f}s")
print(f"Baseline: 2.7805")
print(f"WAL-0: 2.7828")
print(f"Delta: {ppl.item() - 2.7805:+.4f}")
if ppl.item() <= 2.80:
    print("  -> QUALITY PASS: PPL <= 2.80")
else:
    print("  -> QUALITY FAIL: PPL > 2.80")
