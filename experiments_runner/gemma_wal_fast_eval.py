"""Fast comprehensive WAL eval for Gemma-4-31B — samples representative layers."""
import json, math, os, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_DIR = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
DEVICE = "cuda:0"
OUT = Path("/mnt/hf_model_weights/arman/3bit/wal/experiments_runner/results/gemma_wal_fast_eval.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit")
from dwl2_dynamic_route.src.calibrate import calibrate_ladder
from dwl2_dynamic_route.src.route_encoder import encode_routes, decode_routes, rel_mse

SAMPLE_LAYERS = [0, 15, 30, 45, 59]
PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

def encode_cpu(w, lmax=12):
    w_cpu = w.detach().cpu().float()
    row_max = w_cpu.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = w_cpu / row_max
    sample = w_norm.flatten()
    if sample.numel() > 2_000_000:
        sample = sample[torch.randint(0, sample.numel(), (2_000_000,))]
    ladder = calibrate_ladder(sample, l_max=lmax, refine_iters=10, pin_top=True, top_value=1.0, seed="geometric")
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=lmax)
    w_hat_norm = decode_routes(enc, ladder, out_dtype=torch.float32)
    return (w_hat_norm * row_max).to(w.dtype).to(w.device), ladder

def compute_ppl(model, input_ids, max_len=512, stride=256):
    nlls, device = [], next(model.parameters()).device
    input_ids = input_ids.to(device)
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_len, input_ids.size(1))
        if end - begin < stride: break
        with torch.no_grad():
            out = model(input_ids[:, begin:end], labels=input_ids[:, begin:end])
            nlls.append(out.loss.item())
    return float(torch.exp(torch.tensor(nlls).mean()).item())

def main():
    print("=" * 60)
    print("WAL FAST EVAL — Gemma-4-31B-it")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                  device_map=DEVICE, local_files_only=True)
    model.eval()
    print(f"  Loaded on {next(model.parameters()).device}")

    # 2. Baseline PPL
    print("\n[2] Baseline PPL...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in ds['text'] if len(t.strip()) > 0][:100]
    input_ids = tok('\n\n'.join(texts), return_tensors='pt', truncation=True, max_length=2048)['input_ids']
    baseline_ppl = compute_ppl(model, input_ids)
    print(f"  PPL={baseline_ppl:.2f}")

    # 3. Encode sample layers
    print(f"\n[3] Encoding {len(SAMPLE_LAYERS)} layers × {len(PROJS)} projs...")
    results = []
    all_params = dict(model.named_parameters())

    for lmax in [12, 7]:
        for l in SAMPLE_LAYERS:
            for p in PROJS:
                name = f"model.language_model.layers.{l}.{p}.weight"
                if name not in all_params: continue
                w = all_params[name]
                t0 = time.time()
                w_hat, ladder = encode_cpu(w, lmax=lmax)
                dt = time.time() - t0
                re = rel_mse(w.float().cpu(), w_hat.float().cpu()).item()
                results.append({'layer': l, 'proj': p, 'lmax': lmax, 'rel_mse': re, 'encode_s': dt,
                               'shape': list(w.shape)})
                print(f"  L{l} {p}: lmax={lmax} relMSE={re:.6f} ({dt:.1f}s)")

    # 4. Full model encode at lmax=12 + PPL
    print("\n[4] Full model encode lmax=12 + PPL...")
    t0 = time.time()
    for name, w in all_params.items():
        if 'language_model.layers.' in name and 'weight' in name:
            w_hat, _ = encode_cpu(w, lmax=12)
            all_params[name].data.copy_(w_hat)
    encode_time = time.time() - t0
    routed_ppl = compute_ppl(model, input_ids)
    print(f"  Encoded in {encode_time:.1f}s, PPL={routed_ppl:.2f} (baseline={baseline_ppl:.2f}, gap={routed_ppl-baseline_ppl:+.2f})")

    # 5. Frozen vocab test
    print("\n[5] Frozen vocab test...")
    w0 = all_params[f"model.language_model.layers.0.mlp.down_proj.weight"].clone()
    w1 = all_params[f"model.language_model.layers.1.mlp.down_proj.weight"].clone()
    # Encode layer 0
    w0_hat, ladder0 = encode_cpu(w0, lmax=12)
    enc0_before = encode_routes((w0.float().cpu() / w0.float().cpu().abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)),
                                ladder0, l_max=12)
    # Perturb
    w0_edit = w0.clone()
    n = w0_edit.numel() // 10
    idx = torch.randperm(w0_edit.numel())[:n]
    w0_edit.view(-1)[idx] += torch.randn(n) * 0.01
    w0_hat2, _ = encode_cpu(w0_edit, lmax=12)
    enc0_after = encode_routes((w0_edit.float().cpu() / w0_edit.float().cpu().abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)),
                                ladder0, l_max=12)
    # Encode layer 1 with same ladder
    w1_norm = w1.float().cpu() / w1.float().cpu().abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)
    enc1_before = encode_routes(w1_norm, ladder0, l_max=12)
    enc1_after = encode_routes(w1_norm, ladder0, l_max=12)  # should be identical

    target_diff = (enc0_before.digits != enc0_after.digits).float().mean().item()
    non_target_diff = (enc1_before.digits != enc1_after.digits).float().mean().item()

    print(f"  target_diff={target_diff:.4f} non_target_diff={non_target_diff:.4f} frozen={'✓' if non_target_diff==0 else '✗'}")

    # Save
    report = {'model': 'gemma-4-31B-it', 'baseline_ppl': baseline_ppl,
              'routed_ppl': routed_ppl, 'ppl_gap': routed_ppl - baseline_ppl,
              'encode_time_s': encode_time, 'num_layers_encoded': 60,
              'layer_results': results,
              'frozen_vocab': {'target_diff': target_diff, 'non_target_diff': non_target_diff}}
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n✓ {OUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
