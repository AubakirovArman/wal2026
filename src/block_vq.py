from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def _hadamard_matrix(n: int, *, device: torch.device) -> Tensor:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("hadamard transform requires power-of-two block_size")
    mat = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while mat.shape[0] < n:
        mat = torch.cat(
            (
                torch.cat((mat, mat), dim=1),
                torch.cat((mat, -mat), dim=1),
            ),
            dim=0,
        )
    return mat / math.sqrt(float(n))


def _rand_hadamard_matrix(n: int, *, device: torch.device) -> Tensor:
    base = _hadamard_matrix(n, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    perm = torch.randperm(n, generator=generator, device=device)
    signs = torch.where(
        torch.randint(0, 2, (n,), generator=generator, device=device, dtype=torch.int64) > 0,
        torch.ones(n, device=device, dtype=torch.float32),
        -torch.ones(n, device=device, dtype=torch.float32),
    )
    return base[perm] * signs.unsqueeze(1)


def _dct_matrix(n: int, *, device: torch.device) -> Tensor:
    idx = torch.arange(n, device=device, dtype=torch.float32)
    k = idx[:, None]
    mat = torch.cos(math.pi / float(n) * (idx + 0.5)[None, :] * k)
    mat[0] *= math.sqrt(1.0 / float(n))
    if n > 1:
        mat[1:] *= math.sqrt(2.0 / float(n))
    return mat


def _transform_matrix(kind: str, block_size: int, *, device: torch.device) -> Tensor | None:
    if kind == "none":
        return None
    if kind == "dct":
        return _dct_matrix(block_size, device=device)
    if kind == "hadamard":
        return _hadamard_matrix(block_size, device=device)
    if kind == "rand_hadamard":
        return _rand_hadamard_matrix(block_size, device=device)
    if kind == "polar":
        return None
    if kind == "pca":
        return None
    raise ValueError("transform_kind must be one of: none, dct, hadamard, rand_hadamard, polar, pca")


def _polar_transform(blocks: Tensor) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("polar transform expects 2D blocks")
    if blocks.shape[1] < 1 or (blocks.shape[1] & (blocks.shape[1] - 1)) != 0:
        raise ValueError("polar transform requires power-of-two block_size")
    radii = blocks.to(torch.float32)
    angles: list[Tensor] = []
    while radii.shape[1] > 1:
        pair = radii.view(radii.shape[0], -1, 2)
        x = pair[..., 0]
        y = pair[..., 1]
        radii = torch.sqrt(x.square() + y.square())
        angles.append(torch.atan2(y, x) / math.pi)
    return torch.cat([radii] + list(reversed(angles)), dim=1)


def _inverse_polar_transform(blocks: Tensor) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("inverse polar transform expects 2D blocks")
    if blocks.shape[1] < 1 or (blocks.shape[1] & (blocks.shape[1] - 1)) != 0:
        raise ValueError("inverse polar transform requires power-of-two block_size")
    count = int(blocks.shape[1])
    widths = []
    width = 1
    while width < count:
        widths.append(width)
        width *= 2
    current = blocks[:, :1].to(torch.float32)
    offset = 1
    for width in widths:
        angles = blocks[:, offset:offset + width].to(torch.float32)
        offset += width
        current = torch.stack(
            (
                current * torch.cos(math.pi * angles),
                current * torch.sin(math.pi * angles),
            ),
            dim=-1,
        ).reshape(blocks.shape[0], width * 2)
    return current


def _sign_correction_matrix(block_size: int, *, device: torch.device) -> Tensor:
    return _rand_hadamard_matrix(block_size, device=device)


def _pack_sign_bits(signs: Tensor) -> Tensor:
    if signs.ndim != 2:
        raise ValueError("sign packing expects a 2D tensor")
    signs = signs.to(torch.bool)
    words = (signs.shape[1] + 31) // 32
    packed = torch.zeros(signs.shape[0], words, dtype=torch.int32, device=signs.device)
    for word_idx in range(words):
        start = word_idx * 32
        end = min(start + 32, signs.shape[1])
        word = torch.zeros(signs.shape[0], dtype=torch.int64, device=signs.device)
        for bit_idx in range(end - start):
            word = word | (signs[:, start + bit_idx].to(torch.int64) << bit_idx)
        packed[:, word_idx] = word.to(torch.int32)
    return packed


def _unpack_sign_bits(packed: Tensor, width: int) -> Tensor:
    if packed.ndim != 2:
        raise ValueError("sign unpacking expects a 2D tensor")
    out = torch.empty(packed.shape[0], width, dtype=torch.bool, device=packed.device)
    for col in range(width):
        word_idx = col // 32
        bit_idx = col % 32
        word = packed[:, word_idx].to(torch.int64)
        out[:, col] = ((word >> bit_idx) & 1).bool()
    return out


def _fit_pca_transform(sample_blocks: Tensor) -> tuple[Tensor, Tensor]:
    sample_blocks = sample_blocks.to(torch.float32)
    bias = sample_blocks.mean(dim=0, keepdim=True)
    centered = sample_blocks - bias
    cov = centered.t() @ centered / max(int(centered.shape[0]) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    basis = eigvecs[:, order]
    transform = basis.t().contiguous()
    return transform, bias.contiguous()


def _id_storage_bytes(codebook_size: int) -> int:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if codebook_size <= 256:
        return 1
    if codebook_size <= 65_536:
        return 2
    return 4


def _id_storage_dtype(codebook_size: int) -> torch.dtype:
    if codebook_size <= 256:
        return torch.uint8
    if codebook_size <= 32_767:
        return torch.int16
    return torch.int32


def _reshape_blocks(w_norm: Tensor, block_size: int) -> tuple[Tensor, int, int]:
    if w_norm.ndim != 2:
        raise ValueError("expected 2D weight matrix")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    rows, cols = w_norm.shape
    padded_cols = ((cols + block_size - 1) // block_size) * block_size
    if padded_cols != cols:
        w_norm = torch.nn.functional.pad(w_norm, (0, padded_cols - cols), value=0.0)
    return w_norm.view(rows * (padded_cols // block_size), block_size), rows, padded_cols


def _assign_to_codebook(blocks: Tensor, codebook: Tensor, batch_size: int = 16_384) -> tuple[Tensor, Tensor]:
    if blocks.numel() == 0:
        raise ValueError("blocks must be non-empty")
    blocks = blocks.to(torch.float32)
    codebook = codebook.to(device=blocks.device, dtype=torch.float32)
    code_sq = codebook.square().sum(dim=1)
    ids = torch.empty(blocks.shape[0], dtype=torch.int64, device=blocks.device)
    min_dist = torch.empty(blocks.shape[0], dtype=torch.float32, device=blocks.device)
    for start in range(0, blocks.shape[0], batch_size):
        end = min(start + batch_size, blocks.shape[0])
        chunk = blocks[start:end]
        chunk_sq = chunk.square().sum(dim=1, keepdim=True)
        dist = chunk_sq + code_sq.unsqueeze(0) - 2.0 * (chunk @ codebook.t())
        best_dist, best_ids = dist.min(dim=1)
        ids[start:end] = best_ids
        min_dist[start:end] = best_dist
    return ids, min_dist


def _fit_kmeans(
    blocks: Tensor,
    codebook_size: int,
    iters: int,
    batch_size: int = 16_384,
) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("expected 2D blocks")
    if blocks.shape[0] < codebook_size:
        raise ValueError("sample set must be at least as large as codebook_size")
    blocks = blocks.to(torch.float32)
    generator = torch.Generator(device=blocks.device)
    generator.manual_seed(0)
    perm = torch.randperm(blocks.shape[0], generator=generator, device=blocks.device)
    codebook = blocks[perm[:codebook_size]].clone()
    for _ in range(iters):
        ids, _ = _assign_to_codebook(blocks, codebook, batch_size=batch_size)
        counts = torch.bincount(ids, minlength=codebook_size)
        sums = torch.zeros_like(codebook)
        sums.index_add_(0, ids, blocks)
        nonzero = counts > 0
        updated = codebook.clone()
        updated[nonzero] = sums[nonzero] / counts[nonzero].to(sums.dtype).unsqueeze(1)
        empty = (~nonzero).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            reseed = torch.randint(0, blocks.shape[0], (empty.numel(),), generator=generator, device=blocks.device)
            updated[empty] = blocks[reseed]
        codebook = updated
    return codebook


@dataclass
class BlockRVQEncoding:
    stage_ids: tuple[Tensor, ...]
    codebooks: tuple[Tensor, ...]
    stage_value_dims: tuple[int, ...]
    stages_per_split: tuple[int, ...]
    stage_scales: Tensor | None
    residual_correction: str
    residual_signs: Tensor | None
    residual_scale: Tensor | None
    row_scale: Tensor
    block_scale: Tensor | None
    transform_kind: str
    transform_matrix: Tensor | None
    transform_bias: Tensor | None
    product_splits: int
    original_shape: tuple[int, int]
    padded_cols: int
    block_size: int
    sample_rel_mse: float

    @property
    def num_stages(self) -> int:
        return len(self.codebooks)

    @property
    def stage_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.stage_ids[0].shape)

    def reconstruct(self, out_dtype: torch.dtype | None = None) -> Tensor:
        out_dtype = out_dtype or torch.float32
        rows, blocks_per_row = self.stage_shape
        flat_blocks = rows * blocks_per_row
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(zip(self.stage_ids, self.codebooks)):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            recon = _inverse_polar_transform(recon)
        else:
            transform = self.transform_matrix
            if transform is None:
                transform = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform is not None:
                recon = recon @ transform
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        w_norm = recon.view(rows, blocks_per_row * self.block_size)[:, : self.original_shape[1]]
        return (w_norm.to(out_dtype) * self.row_scale.to(out_dtype))

    def storage_bytes(self) -> int:
        total = 0
        for ids, codebook, value_dim in zip(self.stage_ids, self.codebooks, self.stage_value_dims):
            total += ids.numel() * _id_storage_bytes(int(codebook.shape[0]))
            total += int(codebook.shape[0]) * int(value_dim) * 2
        if self.stage_scales is not None:
            total += self.stage_scales.numel() * self.stage_scales.element_size()
        if self.residual_signs is not None:
            total += self.residual_signs.numel() * self.residual_signs.element_size()
        if self.residual_scale is not None:
            total += self.residual_scale.numel() * self.residual_scale.element_size()
        if self.block_scale is not None:
            total += self.block_scale.numel() * self.block_scale.element_size()
        if self.transform_matrix is not None:
            total += self.transform_matrix.numel() * self.transform_matrix.element_size()
        if self.transform_bias is not None:
            total += self.transform_bias.numel() * self.transform_bias.element_size()
        total += self.row_scale.numel() * 2
        return total

    def bits_per_weight(self) -> float:
        rows, cols = self.original_shape
        return 8.0 * self.storage_bytes() / max(rows * cols, 1)


@dataclass
class GroupedBlockRVQEncoding:
    groups: tuple[BlockRVQEncoding, ...]
    row_slices: tuple[tuple[int, int], ...]
    original_shape: tuple[int, int]

    @property
    def sample_rel_mse(self) -> float:
        total_rows = float(sum(group.original_shape[0] for group in self.groups))
        if total_rows <= 0.0:
            return 0.0
        return float(sum(group.sample_rel_mse * group.original_shape[0] for group in self.groups) / total_rows)

    def reconstruct(self, out_dtype: torch.dtype | None = None) -> Tensor:
        parts = [group.reconstruct(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def storage_bytes(self) -> int:
        return sum(group.storage_bytes() for group in self.groups)

    def bits_per_weight(self) -> float:
        rows, cols = self.original_shape
        return 8.0 * self.storage_bytes() / max(rows * cols, 1)


@torch.no_grad()
def encode_block_residual_vq(
    weight: Tensor,
    *,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 2,
    product_splits: int = 1,
    stages_per_split: tuple[int, ...] | None = None,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 131_072,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
) -> BlockRVQEncoding:
    if weight.ndim != 2:
        raise ValueError("expected 2D weight")
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if product_splits < 1 or block_size % product_splits != 0:
        raise ValueError("product_splits must divide block_size and be >= 1")
    if stages_per_split is None:
        split_stage_counts = tuple(int(num_stages) for _ in range(product_splits))
    else:
        split_stage_counts = tuple(int(count) for count in stages_per_split)
        if len(split_stage_counts) != product_splits:
            raise ValueError("stages_per_split must have exactly product_splits entries")
        if any(count < 1 for count in split_stage_counts):
            raise ValueError("each stages_per_split entry must be >= 1")

    row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(torch.float16)
    w_norm = (weight.to(torch.float32) / row_scale.to(torch.float32))
    blocks, rows, padded_cols = _reshape_blocks(w_norm, block_size)
    num_blocks_per_row = padded_cols // block_size
    transform = _transform_matrix(transform_kind, block_size, device=blocks.device)
    transform_bias = None
    if transform_kind == "pca":
        if sample_limit < codebook_size:
            sample_limit = codebook_size
        sample_count = min(int(sample_limit), blocks.shape[0])
        generator = torch.Generator(device=blocks.device)
        generator.manual_seed(0)
        pick_for_pca = torch.randperm(blocks.shape[0], generator=generator, device=blocks.device)[:sample_count]
        transform, transform_bias = _fit_pca_transform(blocks[pick_for_pca])
        blocks = blocks - transform_bias
    if transform_kind == "polar":
        blocks = _polar_transform(blocks)
    elif transform is not None:
        blocks = blocks @ transform.t()

    block_scale = None
    if normalize_blocks == "amax":
        block_scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        blocks = blocks / block_scale
    elif normalize_blocks == "l2":
        block_scale = blocks.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        blocks = blocks / block_scale
    elif normalize_blocks != "none":
        raise ValueError("normalize_blocks must be one of: none, amax, l2")

    residual = blocks.clone()
    if sample_limit < codebook_size:
        sample_limit = codebook_size
    sample_count = min(int(sample_limit), residual.shape[0])
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(0)
    pick = torch.randperm(residual.shape[0], generator=generator, device=residual.device)[:sample_count]
    sample_residual = residual[pick].clone()

    stage_ids: list[Tensor] = []
    codebooks: list[Tensor] = []
    stage_value_dims: list[int] = []

    if product_splits == 1:
        for _ in range(split_stage_counts[0]):
            codebook = _fit_kmeans(sample_residual, codebook_size=codebook_size, iters=kmeans_iters, batch_size=batch_size)
            full_ids, _ = _assign_to_codebook(residual, codebook, batch_size=batch_size)
            residual -= codebook[full_ids]
            sample_ids, _ = _assign_to_codebook(sample_residual, codebook, batch_size=batch_size)
            sample_residual -= codebook[sample_ids]
            stage_ids.append(full_ids.view(rows, num_blocks_per_row).to(_id_storage_dtype(codebook_size)))
            codebooks.append(codebook.to(torch.float16))
            stage_value_dims.append(block_size)
    else:
        sub_dim = block_size // product_splits
        for split, split_stage_count in enumerate(split_stage_counts):
            lo = split * sub_dim
            hi = lo + sub_dim
            residual_slice = residual[:, lo:hi]
            sample_slice = sample_residual[:, lo:hi]
            for _ in range(split_stage_count):
                codebook_slice = _fit_kmeans(sample_slice, codebook_size=codebook_size, iters=kmeans_iters, batch_size=batch_size)
                full_ids, _ = _assign_to_codebook(residual_slice, codebook_slice, batch_size=batch_size)
                residual_slice -= codebook_slice[full_ids]
                sample_ids, _ = _assign_to_codebook(sample_slice, codebook_slice, batch_size=batch_size)
                sample_slice -= codebook_slice[sample_ids]
                codebook_full = torch.zeros(codebook_size, block_size, dtype=torch.float32, device=residual.device)
                codebook_full[:, lo:hi] = codebook_slice
                stage_ids.append(full_ids.view(rows, num_blocks_per_row).to(_id_storage_dtype(codebook_size)))
                codebooks.append(codebook_full.to(torch.float16))
                stage_value_dims.append(sub_dim)
            residual[:, lo:hi] = residual_slice
            sample_residual[:, lo:hi] = sample_slice

    if residual_correction not in {"none", "sign"}:
        raise ValueError("residual_correction must be one of: none, sign")

    stage_scales = None
    full_stage_blocks = [
        codebook[ids.reshape(-1).long()].to(torch.float32)
        for ids, codebook in zip(stage_ids, codebooks)
    ]
    baseline_recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
    for stage in full_stage_blocks:
        baseline_recon_blocks = baseline_recon_blocks + stage

    def _weight_rel_mse(block_recon: Tensor) -> Tensor:
        restored = block_recon
        if block_scale is not None:
            restored = restored * block_scale.to(torch.float32)
        if transform is not None:
            restored = restored @ transform.to(torch.float32)
        if transform_bias is not None:
            restored = restored + transform_bias.to(torch.float32)
        weight_recon = restored.view(rows, num_blocks_per_row * block_size)[:, : int(weight.shape[1])]
        weight_recon = weight_recon * row_scale.to(torch.float32)
        return (weight.to(torch.float32) - weight_recon).square().mean() / weight.to(torch.float32).square().mean().clamp_min(1e-12)

    baseline_weight_loss = _weight_rel_mse(baseline_recon_blocks)
    if calibrate_stage_scales and full_stage_blocks:
        gram = torch.empty(len(full_stage_blocks), len(full_stage_blocks), dtype=torch.float32, device=blocks.device)
        rhs = torch.empty(len(full_stage_blocks), dtype=torch.float32, device=blocks.device)
        target = blocks.to(torch.float32)
        for i, stage_i in enumerate(full_stage_blocks):
            rhs[i] = torch.sum(stage_i * target)
            for j, stage_j in enumerate(full_stage_blocks[i:], start=i):
                value = torch.sum(stage_i * stage_j)
                gram[i, j] = value
                gram[j, i] = value
        damp = 1e-4 * torch.eye(len(full_stage_blocks), device=blocks.device, dtype=torch.float32)
        solved = torch.linalg.solve(gram + damp, rhs)
        candidate_stage_scales = solved.clamp_min(0.0)
        if float(candidate_stage_scales.sum().item()) > 0.0:
            candidate_recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
            for idx, stage in enumerate(full_stage_blocks):
                candidate_recon_blocks = candidate_recon_blocks + stage * candidate_stage_scales[idx]
            candidate_weight_loss = _weight_rel_mse(candidate_recon_blocks)
            if bool(candidate_weight_loss < baseline_weight_loss):
                stage_scales = candidate_stage_scales
    if stage_scales is not None:
        recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
        for idx, stage in enumerate(full_stage_blocks):
            recon_blocks = recon_blocks + stage * stage_scales[idx]
    else:
        recon_blocks = baseline_recon_blocks
    residual_signs = None
    residual_scale = None
    current_weight_loss = _weight_rel_mse(recon_blocks)
    if residual_correction == "sign":
        residual = blocks.to(torch.float32) - recon_blocks
        correction_matrix = _sign_correction_matrix(block_size, device=blocks.device)
        projected_residual = residual @ correction_matrix.t()
        candidate_scale = projected_residual.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        candidate_signs = projected_residual >= 0
        candidate_correction = (candidate_signs.to(torch.float32) * 2.0 - 1.0) * candidate_scale
        candidate_correction = candidate_correction @ correction_matrix.to(torch.float32)
        candidate_weight_loss = _weight_rel_mse(recon_blocks + candidate_correction)
        if bool(candidate_weight_loss < current_weight_loss):
            residual_signs = _pack_sign_bits(candidate_signs)
            residual_scale = candidate_scale.to(torch.float16)
            recon_blocks = recon_blocks + candidate_correction
    sample_rel_mse = float(
        ((blocks[pick].to(torch.float32) - recon_blocks[pick]).square().mean() / blocks[pick].square().mean().clamp_min(1e-12)).item()
    )
    return BlockRVQEncoding(
        stage_ids=tuple(stage_ids),
        codebooks=tuple(codebooks),
        stage_value_dims=tuple(stage_value_dims),
        stages_per_split=split_stage_counts,
        stage_scales=None if stage_scales is None else stage_scales.to(torch.float16),
        residual_correction="none" if residual_signs is None else residual_correction,
        residual_signs=None if residual_signs is None else residual_signs.contiguous(),
        residual_scale=None if residual_scale is None else residual_scale.contiguous(),
        row_scale=row_scale,
        block_scale=None if block_scale is None else block_scale.view(rows, num_blocks_per_row).to(torch.float16),
        transform_kind=transform_kind,
        transform_matrix=None if transform is None or transform_kind != "pca" else transform.to(torch.float16),
        transform_bias=None if transform_bias is None else transform_bias.to(torch.float16),
        product_splits=product_splits,
        original_shape=(int(weight.shape[0]), int(weight.shape[1])),
        padded_cols=padded_cols,
        block_size=block_size,
        sample_rel_mse=sample_rel_mse,
    )


@torch.no_grad()
def encode_grouped_block_residual_vq(
    weight: Tensor,
    *,
    group_rows: int,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 2,
    product_splits: int = 1,
    stages_per_split: tuple[int, ...] | None = None,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 131_072,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
) -> GroupedBlockRVQEncoding:
    if group_rows < 1:
        raise ValueError("group_rows must be >= 1")
    groups: list[BlockRVQEncoding] = []
    row_slices: list[tuple[int, int]] = []
    for start in range(0, int(weight.shape[0]), group_rows):
        end = min(start + group_rows, int(weight.shape[0]))
        groups.append(
            encode_block_residual_vq(
                weight[start:end],
                block_size=block_size,
                codebook_size=codebook_size,
                num_stages=num_stages,
                product_splits=product_splits,
                stages_per_split=stages_per_split,
                normalize_blocks=normalize_blocks,
                transform_kind=transform_kind,
                calibrate_stage_scales=calibrate_stage_scales,
                residual_correction=residual_correction,
                sample_limit=sample_limit,
                kmeans_iters=kmeans_iters,
                batch_size=batch_size,
            )
        )
        row_slices.append((start, end))
    return GroupedBlockRVQEncoding(
        groups=tuple(groups),
        row_slices=tuple(row_slices),
        original_shape=(int(weight.shape[0]), int(weight.shape[1])),
    )


@torch.no_grad()
def sample_row_similarity(stage_ids: tuple[Tensor, ...], n_pairs: int = 1024) -> dict[str, float]:
    if len(stage_ids) < 1:
        raise ValueError("need at least one stage")
    rows, blocks_per_row = stage_ids[0].shape
    combined = torch.zeros(rows, blocks_per_row, dtype=torch.int64, device=stage_ids[0].device)
    radix = 1
    for ids in stage_ids:
        max_id = int(ids.max().item()) + 1
        combined += ids.to(torch.int64) * radix
        radix *= max(max_id, 1)
    generator = torch.Generator(device=combined.device)
    generator.manual_seed(0)
    a = torch.randint(0, rows, (n_pairs,), generator=generator, device=combined.device)
    b = torch.randint(0, rows, (n_pairs,), generator=generator, device=combined.device)
    sim = (combined[a] == combined[b]).float().mean(dim=1).cpu()
    return {
        "mean_sim": float(sim.mean().item()),
        "p50_sim": float(sim.median().item()),
        "p90_sim": float(sim.quantile(0.90).item()),
        "p99_sim": float(sim.quantile(0.99).item()),
    }


@torch.no_grad()
def sample_grouped_row_similarity(groups: tuple[BlockRVQEncoding, ...], n_pairs_per_group: int = 256) -> dict[str, float]:
    sims = []
    weights = []
    for group in groups:
        if group.stage_ids[0].shape[0] < 2:
            continue
        stats = sample_row_similarity(group.stage_ids, n_pairs=n_pairs_per_group)
        sims.append(stats)
        weights.append(group.stage_ids[0].shape[0])
    if not sims:
        return {"mean_sim": 0.0, "p50_sim": 0.0, "p90_sim": 0.0, "p99_sim": 0.0}
    total = float(sum(weights))
    return {
        key: float(sum(stats[key] * w for stats, w in zip(sims, weights)) / total)
        for key in sims[0]
    }


def storage_megabytes(num_bytes: int) -> float:
    return float(num_bytes) / 1e6


def dense_bf16_storage_bytes(weight: Tensor) -> int:
    return int(weight.numel()) * 2


def ideal_id_bits_per_weight(block_size: int, codebook_size: int, num_stages: int) -> float:
    return num_stages * math.ceil(math.log2(max(codebook_size, 2))) / float(block_size)