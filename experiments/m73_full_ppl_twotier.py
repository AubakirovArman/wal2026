#!/usr/bin/env python3
"""M73: Full-model PPL for two-tier uniform quantization.

Tier 1: coarse uniform quantize (K1 levels per column)
Tier 2: residual uniform quantize (K2 levels per column)

Tests:
  - K1=16, K2=16: 8 bits total, 2x compression
  - K1=16, K2=256: 12 bits total, 1.33x compression  
  - K1=32, K2=128: 12 bits total
"""
import torch
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "unsloth/Llama-3.3-70B-Instruct"


def twotier_quantize_all(model, K1, K2):
    """Two-tier uniform quantization for all 2D params."""
    modified = 0
    for name, p in model.named_parameters():
        if len(p.shape) == 2:
            w = p.data.float()
            row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            w_norm = w / row_scale
            
            # Tier 1: coarse
            col_min = w_norm.min(dim=0, keepdim=True)[0]
            col_max = w_norm.max(dim=0, keepdim=True)[0]
            col_range = (col_max - col_min).clamp_min(1e-8)
            scaled1 = (w_norm - col_min) / col_range * (K1 - 1)
            idx1 = torch.round(scaled1).clamp(0, K1 - 1)
            recon1 = idx1.float() / (K1 - 1) * col_range + col_min
            
            # Tier 2: residual
            residual = w_norm - recon1
            r_min = residual.min(dim=0, keepdim=True)[0]
            r_max = residual.max(dim=0, keepdim=True)[0]
            r_range = (r_max - r_min).clamp_min(1e-8)
            scaled2 = (residual - r_min) / r_range * (K2 - 1)
            idx2 = torch.round(scaled2).clamp(0, K2 - 1)
            recon2 = idx2.float() / (K2 - 1) * r_range + r_min
            
            final_recon = (recon1 + recon2) * row_scale
            p.data.copy_(final_recon.to(p.dtype))
            modified += 1
    return modified


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
    
    print("Saving state_dict...")
    t0 = time.time()
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  Saved in {time.time()-t0:.1f}s")
    
    print("\n=== BASELINE ===")
    t0 = time.time()
    baseline_ppl, _, _ = run_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f} ({time.time()-t0:.1f}s)")
    
    configs = [
        (16, 16, "8 bits, 2.0x"),
        (16, 256, "12 bits, 1.33x"),
        (32, 128, "12 bits, 1.33x"),
    ]
    
    print(f"\n{'='*75}")
    print(f"{'K1':>5} | {'K2':>5} | {'Bits':>8} | {'PPL':>10} | {'Delta':>10} | {'Status':>8}")
    print(f"{'-'*75}")
    
    for K1, K2, desc in configs:
        t0 = time.time()
        modified = twotier_quantize_all(model, K1, K2)
        enc_t = time.time() - t0
        
        t0 = time.time()
        ppl, _, _ = run_ppl(model, tokenizer, device)
        inf_t = time.time() - t0
        delta = ppl - baseline_ppl
        status = "PASS" if delta < 0.05 else ("DEGRADE" if delta < 0.5 else "FAIL")
        
        print(f"{K1:>5} | {K2:>5} | {desc:>8} | {ppl:>10.4f} | {delta:>+10.4f} | {status:>8}  (e{enc_t:.1f}s i{inf_t:.1f}s)")
        
        # Restore
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in original_state:
                    p.copy_(original_state[name].to(p.device))
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"{'='*75}")
    print("\nM73 complete.")


if __name__ == "__main__":
    main()
