import torch

from framework.encode import encode_tensor
from wal.v2.format import deserialize_wal_v2, serialize_wal_v2
from wal.v2.grammar import format_wal_text, parse_wal_text
from wal.v2.isa import AtomTable, CoeffTable
from wal.v2.encoder import wal_encode_v2


def test_wal_v2_text_roundtrip_with_tables():
    weights = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
    atoms = AtomTable(torch.tensor([-1.0, -0.25, 0.25, 1.0], dtype=torch.float32))
    coeffs = CoeffTable(torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32))
    prog, _ = wal_encode_v2(weights, atoms, coeffs)

    text = format_wal_text(prog, atoms, coeffs)
    prog2, atoms2, coeffs2 = parse_wal_text(text)

    assert torch.equal(prog.atom_ids, prog2.atom_ids)
    assert torch.equal(prog.coeff_ids, prog2.coeff_ids)
    assert torch.allclose(atoms.values, atoms2.values)
    assert torch.allclose(coeffs.values, coeffs2.values)


def test_wal_v2_binary_roundtrip_requires_row_scales():
    weights = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
    atoms = AtomTable(torch.tensor([-1.0, -0.25, 0.25, 1.0], dtype=torch.float32))
    coeffs = CoeffTable(torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32))
    prog, _ = wal_encode_v2(weights, atoms, coeffs)
    row_scales = torch.ones(weights.shape[0], dtype=torch.float32)

    blob = serialize_wal_v2(prog, atoms, coeffs, row_scales)
    prog2, atoms2, coeffs2, row_scales2, meta = deserialize_wal_v2(blob)

    assert torch.equal(prog.atom_ids, prog2.atom_ids)
    assert torch.equal(prog.coeff_ids, prog2.coeff_ids)
    assert torch.allclose(atoms.values, atoms2.values)
    assert torch.allclose(coeffs.values, coeffs2.values)
    assert torch.allclose(row_scales, row_scales2)
    assert tuple(meta["shape"]) == weights.shape


def test_framework_encode_tensor_uses_v2_tables():
    prog, atoms, coeffs, recon = encode_tensor(torch.linspace(-1.0, 1.0, 16), K=4, C=4, device="cpu")

    assert prog.N == 16
    assert atoms.K == 4
    assert coeffs.C == 4
    assert recon.numel() == 16
