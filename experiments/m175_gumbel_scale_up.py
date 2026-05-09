"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M175 — Gumbel-WAL Scale-Up Test.

Tests Gumbel-WAL on models of increasing size: 10M, 30M, 70M, 100M params.
Compares: dense baseline, post-hoc WAL, Gumbel-WAL fixed atoms, Gumbel-WAL learned coeffs.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


class GumbelWALLinear(nn.Module):
    def __init__(self, in_features, out_features, K=64, C=8):
        super().__init__()
        self.K, self.C = K, C
        self.out_features, self.in_features = out_features, in_features
        self.N = out_features * in_features
        
        self.dense_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        with torch.no_grad():
            w = self.dense_weight.data.reshape(-1)
            atoms = build_l0_atoms(w, K=K, iters=1)
            coeffs = build_coeff_table(w, atoms, C=C, iters=1)
            self.register_buffer('atoms', atoms)
            self.register_buffer('coeffs', coeffs)
            recons = atoms.unsqueeze(1) * coeffs.unsqueeze(0)
            self.register_buffer('recons', recons.reshape(-1))
        
        self.logits = nn.Parameter(torch.zeros(self.N, K * C))
    
    @property
    def weight(self):
        prog = gumbel_softmax(self.logits, temperature=0.5, hard=True)
        w = (prog * self.recons.unsqueeze(0)).sum(dim=-1)
        return w.reshape(self.out_features, self.in_features)
    
    def forward(self, x):
        return F.linear(x, self.weight)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, n_heads=4, d_ff=1024, n_layers=4, use_gumbel=False, K=64, C=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(128, d_model)
        
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        
        LinearClass = GumbelWALLinear if use_gumbel else nn.Linear
        self.head = LinearClass(d_model, vocab_size, K=K, C=C) if use_gumbel else nn.Linear(d_model, vocab_size)
        
        self.use_gumbel = use_gumbel
        self._param_count = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.head(h)
    
    def count_params(self):
        return self._param_count


def train_model(model, device, steps=50, batch_size=8, seq_len=32, vocab_size=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    for step in range(steps):
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


def count_program_entropy(model):
    """Measure program entropy for Gumbel-WAL layers."""
    entropies = []
    for module in model.modules():
        if isinstance(module, GumbelWALLinear):
            probs = module.logits.softmax(dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            entropies.append(entropy)
    return sum(entropies) / len(entropies) if entropies else 0


def main():
    print("=" * 60)
    print("M175 — Gumbel-WAL Scale-Up Test")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    configs = [
        ("10M", 256, 4, 1024, 4),
        ("30M", 512, 6, 2048, 8),
        # ("70M", 768, 8, 3072, 12),  # Skip for speed
        # ("100M", 1024, 8, 4096, 16),  # Skip for speed
    ]
    
    results = {}
    
    for name, d_model, n_layers, d_ff, n_heads in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name} (d={d_model}, L={n_layers}, heads={n_heads})")
        print(f"{'='*50}")
        
        # 1. Dense baseline
        print("\n--- Dense Baseline ---")
        model_dense = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_gumbel=False).to(device)
        print(f"  Params: {model_dense.count_params()/1e6:.1f}M")
        
        start = time.time()
        losses_dense = train_model(model_dense, device, steps=30)
        time_dense = time.time() - start
        print(f"  Final loss: {losses_dense[-1]:.4f}, Time: {time_dense:.1f}s")
        
        # 2. Post-hoc WAL
        print("\n--- Post-hoc WAL ---")
        model_wal = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_gumbel=False).to(device)
        # Encode head to WAL
        with torch.no_grad():
            w = model_wal.head.weight.data
            atoms = build_l0_atoms(w.reshape(-1), K=64, iters=1)
            coeffs = build_coeff_table(w.reshape(-1), atoms, C=8, iters=1)
            _, recon = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=65_536)
            model_wal.head.weight.data = recon.reshape(w.shape).to(w.dtype)
        
        start = time.time()
        losses_wal = train_model(model_wal, device, steps=30)
        time_wal = time.time() - start
        print(f"  Final loss: {losses_wal[-1]:.4f}, Time: {time_wal:.1f}s")
        
        # 3. Gumbel-WAL
        print("\n--- Gumbel-WAL ---")
        model_gumbel = TinyTransformer(10000, d_model, n_heads, d_ff, n_layers, use_gumbel=True, K=64, C=8).to(device)
        print(f"  Params: {model_gumbel.count_params()/1e6:.1f}M")
        
        start = time.time()
        losses_gumbel = train_model(model_gumbel, device, steps=30)
        time_gumbel = time.time() - start
        entropy = count_program_entropy(model_gumbel)
        print(f"  Final loss: {losses_gumbel[-1]:.4f}, Time: {time_gumbel:.1f}s, Entropy: {entropy:.4f}")
        
        results[name] = {
            "dense_final": losses_dense[-1],
            "dense_time": time_dense,
            "wal_final": losses_wal[-1],
            "wal_time": time_wal,
            "gumbel_final": losses_gumbel[-1],
            "gumbel_time": time_gumbel,
            "gumbel_entropy": entropy,
            "params_M": model_dense.count_params() / 1e6,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':>8} {'Dense':>10} {'WAL':>10} {'Gumbel':>10} {'G/D':>8}")
    print("-" * 50)
    for name, r in results.items():
        gd = r['gumbel_final'] / max(r['dense_final'], 1e-10)
        print(f"{name:>8} {r['dense_final']:>10.4f} {r['wal_final']:>10.4f} {r['gumbel_final']:>10.4f} {gd:>8.2f}")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m175_gumbel_scale_up.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
