"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M167 v2 — STE/Gumbel Program IDs (differentiable WAL).

Tests whether we can learn atom_ids directly via Gumbel-Softmax.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=2, d_ff=256, n_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(32, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.head(h)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Gumbel-Softmax with optional hard STE."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim=-1)
    
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft  # STE
    else:
        y = y_soft
    return y


class GumbelWALLinear(nn.Module):
    def __init__(self, in_features, out_features, K=32, C=4):
        super().__init__()
        self.K, self.C = K, C
        self.out_features, self.in_features = out_features, in_features
        self.N = out_features * in_features
        
        # Initialize from dense weight
        self.dense_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Build atoms and coeffs from initial dense weight
        with torch.no_grad():
            w = self.dense_weight.data.reshape(-1)
            atoms = build_l0_atoms(w, K=K, iters=1)
            coeffs = build_coeff_table(w, atoms, C=C, iters=1)
            self.register_buffer('atoms', atoms)
            self.register_buffer('coeffs', coeffs)
            
            # Pre-compute all reconstructions [N, K*C]
            recons = atoms.unsqueeze(1) * coeffs.unsqueeze(0)  # [K, C]
            self.register_buffer('recons', recons.reshape(-1))  # [K*C]
        
        # Learnable logits for program IDs [N, K*C]
        self.logits = nn.Parameter(torch.zeros(self.N, K * C))
    
    @property
    def weight(self):
        prog = gumbel_softmax(self.logits, temperature=0.5, hard=True)
        w = (prog * self.recons.unsqueeze(0)).sum(dim=-1)
        return w.reshape(self.out_features, self.in_features)
    
    def forward(self, x):
        return F.linear(x, self.weight)


def replace_linear_with_gumbel(model, K=32, C=4):
    """Replace all nn.Linear with GumbelWALLinear."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            gumbel = GumbelWALLinear(module.in_features, module.out_features, K=K, C=C)
            gumbel.dense_weight.data = module.weight.data.clone()
            if module.bias is not None:
                gumbel.bias = nn.Parameter(module.bias.data.clone())
            setattr(parent, parts[-1], gumbel)
    return model


def train_step(model, data, target, optimizer):
    optimizer.zero_grad()
    out = model(data)
    loss = nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    print("=" * 60)
    print("M167 v2 — STE/Gumbel Program IDs")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Baseline
    model_dense = TinyTransformer(vocab_size=1000, d_model=128, n_heads=2, d_ff=256, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model_dense.parameters(), lr=1e-3)
    
    print("\n--- Baseline training ---")
    baseline = []
    for step in range(30):
        data = torch.randint(0, 1000, (8, 32), device=device)
        target = torch.randint(0, 1000, (8, 32), device=device)
        loss = train_step(model_dense, data, target, optimizer)
        baseline.append(loss)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}")
    
    # Gumbel-WAL
    print("\n--- Gumbel-WAL training ---")
    model_gumbel = TinyTransformer(vocab_size=1000, d_model=128, n_heads=2, d_ff=256, n_layers=1).to(device)
    replace_linear_with_gumbel(model_gumbel, K=32, C=4)
    model_gumbel = model_gumbel.to(device)
    optimizer2 = torch.optim.Adam(model_gumbel.parameters(), lr=1e-3)
    
    gumbel = []
    for step in range(30):
        data = torch.randint(0, 1000, (8, 32), device=device)
        target = torch.randint(0, 1000, (8, 32), device=device)
        loss = train_step(model_gumbel, data, target, optimizer2)
        gumbel.append(loss)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}")
    
    print(f"\nBaseline final: {baseline[-1]:.4f}")
    print(f"Gumbel final:   {gumbel[-1]:.4f}")
    print(f"Gumbel viable:  {gumbel[-1] < baseline[0] * 1.5}")
    
    results = {
        'baseline': baseline,
        'gumbel': gumbel,
        'baseline_final': baseline[-1],
        'gumbel_final': gumbel[-1],
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m167_ste_gumbel.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
