"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M75: WAL v1 full 70B PPL + round-trip verification.

WAL v1 uses same encode as v2 (12 bits/weight) but adds hierarchical atom definitions.
Expected PPL: same as v2 (~2.778).
"""
import torch
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from wal.v1 import (
    build_l0_atoms, build_coeff_table, wal_encode_v1,
    build_hierarchical_atoms, wal_decode_v1, apply_row_scale,
    assemble, disassemble,
)

model_name = "unsloth/Llama-3.3-70B-Instruct"
K_ATOMS = 256
C_COEFFS = 16
KMEANS_ITERS = 5
SAMPLE_SIZE = 1_000_000


def run_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = 2048
    stride = 512
    max_samples = 16
    nlls = []
    prev_end_loc = 0
    num_steps = 0
    for begin_loc in range(0, seq_len, stride):
        if num_steps >= max_samples:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        num_steps += 1
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item(), num_steps, end_loc


def main():
    print(f"Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    
    print("\n=== BASELINE PPL ===")
    t0 = time.time()
    baseline_ppl, _, _ = run_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f} ({time.time()-t0:.1f}s)")
    
    # Encode all layers with WAL v1
    print("\n=== WAL v1 ENCODE ===")
    t0_total = time.time()
    stats = {"encoded": 0, "total_weights": 0, "l1_atoms": 0}
    
    for idx, (name, param) in enumerate(list(model.named_parameters())):
        if len(param.shape) != 2:
            continue
        if "embed_tokens" in name or "lm_head" in name:
            continue
        
        w = param.data.float()
        param_device = param.device
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        flat = w_norm.reshape(-1)
        
        # Build atoms and coeffs
        atoms = build_l0_atoms(flat, K_ATOMS, KMEANS_ITERS, device=param_device)
        coeffs = build_coeff_table(flat, atoms, C_COEFFS, KMEANS_ITERS, device=param_device)
        
        # Encode
        prog, recon = wal_encode_v1(flat, atoms, coeffs, batch=1_048_576, device=param_device)
        
        # Build hierarchical atoms
        atom_table = build_hierarchical_atoms(atoms, prog, max_l1=64)
        stats["l1_atoms"] += (atom_table.K_total - atom_table.K0)
        
        # Verify decode (fast path) — compare raw reconstructions
        recon_check = wal_decode_v1(prog, atom_table, coeffs, use_hierarchical=False)
        max_diff = (recon_check - recon).abs().max()
        assert max_diff < 1e-4, f"Decode mismatch: {max_diff}"
        
        # Apply recon
        w_hat = recon.to(param_device).reshape(w.shape) * row_scale.to(param_device)
        param.data.copy_(w_hat.to(param.dtype))
        
        stats["encoded"] += 1
        stats["total_weights"] += flat.numel()
        
        del atoms, coeffs, prog, recon, atom_table, recon_check
        if idx % 20 == 0:
            torch.cuda.empty_cache()
        
        if stats["encoded"] % 50 == 0:
            print(f"  Encoded {stats['encoded']} params...")
    
    encode_time = time.time() - t0_total
    print(f"\nEncode done in {encode_time:.0f}s")
    print(f"  Encoded: {stats['encoded']} layers, {stats['total_weights']:,} weights")
    print(f"  L1 atoms created: {stats['l1_atoms']}")
    
    # PPL
    print("\n=== WAL v1 PPL ===")
    t0 = time.time()
    ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
    print(f"WAL v1 PPL: {ppl:.4f}  (delta: {ppl - baseline_ppl:+.4f})")
    print(f"Inference time: {time.time()-t0:.1f}s")
    
    if ppl <= 2.80:
        print("  -> QUALITY PASS")
    else:
        print("  -> QUALITY FAIL")
    
    print(f"\n{'='*60}")
    print("M75 complete.")


if __name__ == "__main__":
    main()
