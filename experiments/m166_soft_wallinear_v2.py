#!/usr/bin/env python3
"""M166 v2 — Soft-WALLinear Small Model (synthetic training).

Trains a tiny transformer with WAL-encoded weights to test differentiable program space.
"""
import torch, torch.nn as nn, math, json, sys
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


def wal_encode_linear(linear, K=32, C=4):
    """Replace linear weight with WAL-encoded version."""
    w = linear.weight.data
    atoms = build_l0_atoms(w.reshape(-1), K=K, iters=1)
    coeffs = build_coeff_table(w.reshape(-1), atoms, C=C, iters=1)
    _, recon = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=65_536)
    linear.weight.data = recon.reshape(w.shape).to(w.dtype)
    return atoms, coeffs


def train_step(model, data, target, optimizer):
    optimizer.zero_grad()
    out = model(data)
    loss = nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    print("=" * 60)
    print("M166 v2 — Soft-WALLinear Small Model")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create tiny model
    model = TinyTransformer(vocab_size=1000, d_model=128, n_heads=2, d_ff=256, n_layers=1).to(device)
    
    # Baseline training
    print("\n--- Baseline training (dense) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    baseline_losses = []
    for step in range(50):
        data = torch.randint(0, 1000, (8, 32), device=device)
        target = torch.randint(0, 1000, (8, 32), device=device)
        loss = train_step(model, data, target, optimizer)
        baseline_losses.append(loss)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}")
    
    # WAL-encode all linear layers
    print("\n--- WAL-encoding all linear layers ---")
    wal_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            atoms, coeffs = wal_encode_linear(module, K=32, C=4)
            wal_info[name] = {'K': 32, 'C': 4, 'mse': ((module.weight.data - module.weight.data)**2).mean().item()}
            print(f"  {name}: encoded")
    
    # WAL training
    print("\n--- Training with WAL weights ---")
    wal_losses = []
    for step in range(50):
        data = torch.randint(0, 1000, (8, 32), device=device)
        target = torch.randint(0, 1000, (8, 32), device=device)
        loss = train_step(model, data, target, optimizer)
        wal_losses.append(loss)
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}")
    
    # Summary
    print(f"\nBaseline final: {baseline_losses[-1]:.4f}")
    print(f"WAL final:      {wal_losses[-1]:.4f}")
    print(f"WAL improves:   {wal_losses[-1] < baseline_losses[-1]}")
    
    results = {
        'baseline_losses': baseline_losses,
        'wal_losses': wal_losses,
        'baseline_final': baseline_losses[-1],
        'wal_final': wal_losses[-1],
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m166_soft_wallinear.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
