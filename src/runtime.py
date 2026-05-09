"""Runtime linear layers that reconstruct weights from a route codebook.

PackedIDRouteLinear stores:
  ids        : int32[N, K]       — codebook index per weight position
  codebook_w : fp16[M, L_max]    — signed scales for each unique route
                                    (= digits[i,l] * ladder[l], folded into one fp16
                                     so decode is just a gather + sum)
  row_scale  : fp16[N, 1]        — per-row scale
  bias       : optional fp16[N]

Decode: w[n,k] = row_scale[n,0] * codebook_w[ids[n,k], :].sum()

Then y = x @ w.T + bias.  We compute w on the fly per forward into fp16 via a
single gather, then call F.linear.  This is the reference path; a fused kernel
lives in a separate module (TODO M5).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .block_vq import BlockRVQEncoding, GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
from .calibrate import calibrate_ladder
from .codebook import build_codebook
from .full_layer_tiled_runtime import (
    build_grouped_local_hotprefix_plan,
    build_grouped_local_plan,
    full_layer_grouped_local_hotprefix_matmul,
    full_layer_grouped_local_matmul,
)
from .grouped_runtime import GroupedLocalRouteLinear
from .route_encoder import encode_routes, rel_mse
from .triton_id_matmul import id_route_linear_matmul, rvq_group_linear_matmul
from .triton_stage_local_hot_cold import stage_local_hot_cold_matmul, stage_local_hot_palette_b2_matmul, stage_local_hot_palette_matmul

TARGET_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class PackedIDRouteLinear(nn.Module):
    def __init__(
        self,
        ids: Tensor,              # int16[N,K] or int32[N,K]
        codebook_w: Tensor,       # fp16[M, L_max] with already-signed scales
        row_scale: Tensor,        # fp16[N,1]
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        # store ids as int16 if codebook fits to halve bandwidth vs int32
        if codebook_w.shape[0] <= 32767 and ids.dtype != torch.int16:
            ids = ids.to(torch.int16)
        self.register_buffer("ids", ids.contiguous())
        # codebook_sum is the per-route scalar: sum_l (digit[l] * ladder[l])
        # single fp16 per route → M*2 bytes total (tiny, L2-cacheable)
        self.register_buffer("codebook_sum", codebook_w.sum(dim=-1).contiguous().to(torch.float16))
        self.register_buffer("row_scale", row_scale.to(torch.float16).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16).contiguous())
        else:
            self.bias = None
        self.out_features = ids.shape[0]
        self.in_features = ids.shape[1]

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,           # int32[N,K]
        codebook_digits: Tensor,   # int8[M, L_max]
        ladder: Tensor,        # fp32[L_max]
        row_scale: Tensor,     # fp16 or fp32[N,1]
        bias: Tensor | None = None,
        **_: object,
    ) -> "PackedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias)

    def reconstruct_weight(self) -> Tensor:
        # gather per-position route sums, multiply by row scale → fp16 weight matrix
        w = self.codebook_sum[self.ids.long()]         # fp16[N,K]
        w = w * self.row_scale                         # broadcast over K
        return w

    def forward(self, x: Tensor) -> Tensor:
        w = self.reconstruct_weight()
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class FusedIDRouteLinear(PackedIDRouteLinear):
    def forward(self, x: Tensor) -> Tensor:
        out = id_route_linear_matmul(x, self.ids, self.codebook_sum, self.row_scale)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class CachedPackedIDRouteLinear(PackedIDRouteLinear):
    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        max_cache_bytes: int = 128 * 2**20,
    ) -> None:
        super().__init__(ids, codebook_w, row_scale, bias)
        self.max_cache_bytes = max(int(max_cache_bytes), 0)
        self._cached_weight: Tensor | None = None
        self._cached_weight_device: torch.device | None = None
        self._cached_weight_dtype: torch.dtype | None = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_skip_count = 0

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        max_cache_bytes: int = 128 * 2**20,
        **_: object,
    ) -> "CachedPackedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, max_cache_bytes=max_cache_bytes)

    def clear_cache(self) -> None:
        self._cached_weight = None
        self._cached_weight_device = None
        self._cached_weight_dtype = None

    def _weight_for(self, x: Tensor) -> Tensor:
        weight_bytes = self.out_features * self.in_features * x.element_size()
        if self.max_cache_bytes > 0 and weight_bytes > self.max_cache_bytes:
            self.cache_skip_count += 1
            return self.reconstruct_weight().to(x.dtype).contiguous()
        if (
            self._cached_weight is None
            or self._cached_weight_device != x.device
            or self._cached_weight_dtype != x.dtype
        ):
            self._cached_weight = self.reconstruct_weight().to(x.dtype).contiguous()
            self._cached_weight_device = x.device
            self._cached_weight_dtype = x.dtype
            self.cache_miss_count += 1
        else:
            self.cache_hit_count += 1
        return self._cached_weight

    def forward(self, x: Tensor) -> Tensor:
        weight = self._weight_for(x)
        return F.linear(x, weight, self.bias.to(x.dtype) if self.bias is not None else None)


class EagerBf16Linear(nn.Linear):
    """Materialize the route-decoded weight once and reuse the standard nn.Linear path.

    Inherits from nn.Linear so HuggingFace accelerate / torch.compile / fused
    attention dispatchers see an identical layer signature to the baseline.
    Storage on disk stays 3-bit (codebook), VRAM holds a single bf16 weight
    exactly like the baseline nn.Linear.
    """

    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        out_features = int(ids.shape[0])
        in_features = int(ids.shape[1])
        device = ids.device
        # Skip the default Parameter init (we overwrite it); use empty meta-style.
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        # Build weight directly in target dtype to avoid an fp32 transient.
        codebook_sum = codebook_w.sum(dim=-1).to(target_dtype)
        weight = (codebook_sum[ids.long()] * row_scale.to(target_dtype)).contiguous()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(target_dtype).contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
        **_: object,
    ) -> "EagerBf16Linear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, target_dtype=target_dtype)

    def reconstruct_weight(self) -> Tensor:
        return self.weight.detach().float()


class EagerBlockRVQLinear(nn.Linear):
    """Materialize a block-RVQ approximation once and reuse the standard nn.Linear path."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(weight.to(target_dtype).contiguous(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(target_dtype).contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def reconstruct_weight(self) -> Tensor:
        return self.weight.detach().float()


class PackedBlockRVQGroup(nn.Module):
    def __init__(self, enc: BlockRVQEncoding) -> None:
        super().__init__()
        self.block_size = int(enc.block_size)
        self.in_features = int(enc.original_shape[1])
        self.out_features = int(enc.original_shape[0])
        self.padded_cols = int(enc.padded_cols)
        self.num_stages = len(enc.stage_ids)
        # M21: stages live in split-major order. We keep stages_per_split so
        # callers can request "drop last residual stage uniformly across splits".
        self.product_splits = int(enc.product_splits)
        self.stages_per_split = tuple(int(x) for x in enc.stages_per_split)
        self.transform_kind = enc.transform_kind
        for idx, ids in enumerate(enc.stage_ids):
            self.register_buffer(f"stage_ids_{idx}", ids.contiguous())
        for idx, codebook in enumerate(enc.codebooks):
            self.register_buffer(f"codebook_{idx}", codebook.contiguous().to(torch.float16))
        if enc.stage_scales is not None:
            self.register_buffer("stage_scales", enc.stage_scales.to(torch.float16).contiguous())
        else:
            self.stage_scales = None
        self.residual_correction = enc.residual_correction
        if enc.residual_signs is not None:
            self.register_buffer("residual_signs", enc.residual_signs.contiguous())
        else:
            self.residual_signs = None
        if enc.residual_scale is not None:
            self.register_buffer("residual_scale", enc.residual_scale.to(torch.float16).contiguous())
        else:
            self.residual_scale = None
        self.register_buffer("row_scale", enc.row_scale.to(torch.float16).contiguous())
        if enc.block_scale is not None:
            self.register_buffer("block_scale", enc.block_scale.to(torch.float16).contiguous())
        else:
            self.block_scale = None
        if enc.transform_matrix is not None:
            self.register_buffer("transform_matrix", enc.transform_matrix.to(torch.float16).contiguous())
        else:
            self.transform_matrix = None
        if enc.transform_bias is not None:
            self.register_buffer("transform_bias", enc.transform_bias.to(torch.float16).contiguous())
        else:
            self.transform_bias = None
        self._triton_stage_ids: Tensor | None = None
        self._triton_codebooks: Tensor | None = None
        self._triton_cache_device: torch.device | None = None
        self._triton_stage_local_hot_ids: Tensor | None = None
        self._triton_stage_local_full_codebooks: Tensor | None = None
        self._triton_stage_local_hot_codebooks: Tensor | None = None
        self._triton_stage_local_hot_lut: Tensor | None = None
        self._triton_stage_local_b1_stage_ids: Tensor | None = None
        self._triton_stage_local_b1_full_codebooks: Tensor | None = None
        self._triton_stage_local_b1_hot_ids: Tensor | None = None
        self._triton_stage_local_b1_hot_codebooks: Tensor | None = None
        self._triton_stage_local_b1_row_scale: Tensor | None = None
        self._triton_stage_local_b1_cache_device: torch.device | None = None
        self._triton_stage_local_b1_cache_dtype: torch.dtype | None = None
        self._triton_stage_local_b1_cache_topk: int | None = None
        self._triton_stage_local_b1_cache_score_mode: str | None = None
        self._triton_stage_local_b1_cache_min_stage_share: float | None = None
        self._triton_stage_local_b1_cache_score_threshold_ratio: float | None = None
        self._triton_stage_local_cache_device: torch.device | None = None
        self._triton_stage_local_cache_dtype: torch.dtype | None = None
        self._triton_stage_local_cache_topk: int | None = None
        self._triton_stage_local_cache_score_mode: str | None = None
        self._triton_stage_local_cache_min_stage_share: float | None = None
        self._triton_stage_local_cache_score_threshold_ratio: float | None = None
        # M20 fast-path caches: built lazily on first call of reconstruct_weight_fast
        self._fast_codebooks_cached: list[Tensor] | None = None
        self._fast_row_scale_cached: Tensor | None = None
        self._fast_cache_dtype: torch.dtype | None = None
        self._fast_cache_device: torch.device | None = None
        # M21: variable-stage decoding. effective_stages_per_split controls how
        # many residual stages we sum *per split*. Range: [1, max(stages_per_split)].
        # Defaults to max(stages_per_split) (no change in behavior).
        self.effective_stages_per_split: int = max(self.stages_per_split) if self.stages_per_split else self.num_stages
        # Cached active sub-stage indices, rebuilt lazily when effective_stages_per_split changes.
        self._active_stage_indices: tuple[int, ...] | None = None
        # M23: stage-local hot/cold codebook caches.
        self._hot_codebooks_cached: list[Tensor | None] | None = None
        self._hot_top_ids_cached: list[Tensor | None] | None = None
        self._hot_id_to_slot_cached: list[Tensor | None] | None = None
        self._hot_positions_cached: list[Tensor | None] | None = None
        self._hot_slots_cached: list[Tensor | None] | None = None
        self._cold_positions_cached: list[Tensor | None] | None = None
        self._cold_ids_cached: list[Tensor | None] | None = None
        self._hot_cache_dtype: torch.dtype | None = None
        self._hot_cache_device: torch.device | None = None
        self._hot_cache_topk: int | None = None
        self._hot_cache_score_mode: str | None = None
        self._hot_cache_min_stage_share: float | None = None
        self._hot_cache_score_threshold_ratio: float | None = None
        self._hot_stage_share_cached: list[float] | None = None
        # M26 stage 1: cached recon buffer reused by reconstruct_weight_hot_v2.
        self._hot_recon_buffer_cached: Tensor | None = None
        self._hot_recon_buffer_dtype: torch.dtype | None = None
        self._hot_recon_buffer_device: torch.device | None = None

    def _stage_tensors(self) -> list[tuple[Tensor, Tensor]]:
        return [
            (getattr(self, f"stage_ids_{idx}"), getattr(self, f"codebook_{idx}"))
            for idx in range(self.num_stages)
        ]

    def _compute_active_stage_indices(self) -> tuple[int, ...]:
        """M21: return sub-stage indices that should be summed under
        ``effective_stages_per_split``. Stages are split-major:
        for product_splits=ps and stages_per_split=(s,)*ps, the layout is
        [split0_stage0..s-1, split1_stage0..s-1, ...]. We keep the first
        ``effective_stages_per_split`` stages of every split.
        """
        if self._active_stage_indices is not None:
            return self._active_stage_indices
        keep = max(1, int(self.effective_stages_per_split))
        idxs: list[int] = []
        offset = 0
        for split_count in self.stages_per_split:
            local_keep = min(keep, int(split_count))
            for j in range(local_keep):
                idxs.append(offset + j)
            offset += int(split_count)
        if not idxs:
            idxs = [0]
        self._active_stage_indices = tuple(idxs)
        return self._active_stage_indices

    def set_effective_stages_per_split(self, k: int) -> None:
        """Set per-split active stage count and invalidate the cached index list."""
        self.effective_stages_per_split = int(k)
        self._active_stage_indices = None

    def reconstruct_weight(self, out_dtype: torch.dtype) -> Tensor:
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        flat_blocks = int(rows) * int(blocks_per_row)
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            from .block_vq import _sign_correction_matrix, _unpack_sign_bits

            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device).to(torch.float32)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            from .block_vq import _inverse_polar_transform

            recon = _inverse_polar_transform(recon)
        else:
            transform_matrix = self.transform_matrix
            if transform_matrix is None and self.transform_kind != "none":
                from .block_vq import _transform_matrix

                transform_matrix = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform_matrix is not None:
                recon = recon @ transform_matrix.to(torch.float32)
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        weight = recon.view(rows, blocks_per_row * self.block_size)[:, : self.in_features]
        return (weight.to(out_dtype) * self.row_scale.to(out_dtype))

    def supports_fast_reconstruct(self) -> bool:
        """True for the plain RVQ case where bf16 accumulation suffices.

        Skips fp32 casts, torch.zeros allocation, and per-call dtype conversions.
        """
        return (
            self.block_scale is None
            and self.transform_kind == "none"
            and self.transform_matrix is None
            and self.transform_bias is None
            and self.residual_correction == "none"
            and self.residual_signs is None
            and self.residual_scale is None
        )

    def _build_fast_cache(self, out_dtype: torch.dtype, device: torch.device) -> None:
        """Build cached out_dtype codebooks (with stage_scales merged) and bf16 row_scale.

        Note: ids are NOT pre-cast to int64. Doing so would multiply the on-device
        ids cache size by 8x (uint8 -> int64) and dominate VRAM cost. Instead we
        rely on inline `.long()` of a small contiguous tensor which is cheap.
        """
        codebooks_cached = []
        stage_scales = self.stage_scales
        for idx in range(self.num_stages):
            cb = getattr(self, f"codebook_{idx}").to(out_dtype)
            if stage_scales is not None:
                cb = cb * stage_scales[idx].to(out_dtype)
            codebooks_cached.append(cb.contiguous())
        self._fast_codebooks_cached = codebooks_cached
        self._fast_row_scale_cached = self.row_scale.to(out_dtype).contiguous()
        self._fast_cache_dtype = out_dtype
        self._fast_cache_device = device

    def _build_stage_hot_cache(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> None:
        """Build stage-local hot/cold codebook caches.

        Hot ids are chosen from the current encoding, not an external JSON, so
        the cache stays aligned with the actual per-run codebook assignment.

        Note: exact activation norms from M23 are not available inside the
        runtime surface. `stage_influence_proxy` therefore uses the same static
        stage-local proxy as M23 without the activation term:
        `row_scale * codebook_norm * occurrence_mass`.
        """
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        row_weights = self.row_scale.to(torch.float32).abs().reshape(int(rows), 1).expand(-1, int(blocks_per_row)).reshape(-1)
        hot_codebooks: list[Tensor | None] = []
        hot_top_ids_list: list[Tensor | None] = []
        hot_luts: list[Tensor | None] = []
        hot_positions: list[Tensor | None] = []
        hot_slots_list: list[Tensor | None] = []
        cold_positions: list[Tensor | None] = []
        cold_ids_list: list[Tensor | None] = []
        hot_stage_shares: list[float] = []
        for idx in range(self.num_stages):
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            full_codebook = self._fast_codebooks_cached[idx]
            codebook_size = int(full_codebook.shape[0])
            topk = min(max(int(hot_topk), 1), codebook_size)
            scores = torch.zeros(codebook_size, dtype=torch.float32, device=device)
            if score_mode == "count":
                scores.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
            elif score_mode in {"row_scale_norm", "stage_influence_proxy"}:
                codebook_norm = full_codebook.to(torch.float32).norm(dim=-1)
                scores.scatter_add_(0, ids, row_weights * codebook_norm.index_select(0, ids))
            else:
                raise ValueError(f"unsupported hot score_mode: {score_mode}")
            sorted_ids = torch.argsort(scores, descending=True)
            sorted_scores = scores.index_select(0, sorted_ids)
            if float(score_threshold_ratio) > 0.0 and sorted_scores.numel() > 0 and float(sorted_scores[0].item()) > 0.0:
                threshold = float(score_threshold_ratio) * float(sorted_scores[0].item())
                selected = sorted_ids[sorted_scores >= threshold]
                if selected.numel() == 0:
                    selected = sorted_ids[:1]
                top_ids = selected[:topk]
            else:
                top_ids = sorted_ids[:topk]
            total_score = float(scores.sum().item())
            hot_score = float(scores.index_select(0, top_ids).sum().item())
            hot_share = hot_score / max(total_score, 1e-12)
            hot_stage_shares.append(hot_share)
            if hot_share < float(min_stage_share):
                hot_codebooks.append(None)
                hot_top_ids_list.append(None)
                hot_luts.append(None)
                hot_positions.append(None)
                hot_slots_list.append(None)
                cold_positions.append(None)
                cold_ids_list.append(None)
                continue
            hot_codebook = full_codebook.index_select(0, top_ids).contiguous()
            hot_top_ids_list.append(top_ids.contiguous())
            id_to_slot = torch.full((codebook_size,), -1, dtype=torch.int16, device=device)
            id_to_slot[top_ids] = torch.arange(top_ids.numel(), dtype=torch.int16, device=device)
            hot_slots = id_to_slot.index_select(0, ids)
            hot_mask = hot_slots >= 0
            hot_pos = torch.nonzero(hot_mask, as_tuple=False).reshape(-1).contiguous()
            cold_pos = torch.nonzero(~hot_mask, as_tuple=False).reshape(-1).contiguous()
            hot_codebooks.append(hot_codebook)
            hot_luts.append(id_to_slot)
            hot_positions.append(hot_pos)
            hot_slots_list.append(hot_slots.index_select(0, hot_pos).to(torch.int64).contiguous())
            cold_positions.append(cold_pos)
            cold_ids_list.append(ids.index_select(0, cold_pos).contiguous())
        self._hot_codebooks_cached = hot_codebooks
        self._hot_top_ids_cached = hot_top_ids_list
        self._hot_id_to_slot_cached = hot_luts
        self._hot_positions_cached = hot_positions
        self._hot_slots_cached = hot_slots_list
        self._cold_positions_cached = cold_positions
        self._cold_ids_cached = cold_ids_list
        self._hot_cache_dtype = out_dtype
        self._hot_cache_device = device
        self._hot_cache_topk = int(hot_topk)
        self._hot_cache_score_mode = score_mode
        self._hot_cache_min_stage_share = float(min_stage_share)
        self._hot_cache_score_threshold_ratio = float(score_threshold_ratio)
        self._hot_stage_share_cached = hot_stage_shares

    def _build_hot_cache(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> None:
        self._build_stage_hot_cache(
            out_dtype,
            device,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )

    def reconstruct_weight_fast(self, out_dtype: torch.dtype) -> Tensor:
        """Allocation-light reconstruction for plain RVQ groups.

        Eliminates: torch.zeros allocation, fp32 codebook casts, fp32 accumulation,
        per-call row_scale dtype cast, per-call stage_scales dtype cast.
        Keeps: inline ids.long() cast (cheap; pre-caching int64 ids would 8x VRAM).
        Accumulates directly in out_dtype (bf16) which is sufficient when
        codebook_size <= 256 and num_stages <= 4.
        """
        if not self.supports_fast_reconstruct():
            return self.reconstruct_weight(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        cb_list = self._fast_codebooks_cached
        # M21: build the split-aware list of active sub-stage indices.
        active_idx = self._compute_active_stage_indices()
        first = active_idx[0]
        ids0 = getattr(self, f"stage_ids_{first}").reshape(-1).to(torch.int64)
        recon = cb_list[first].index_select(0, ids0)
        for idx in active_idx[1:]:
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            recon.add_(cb_list[idx].index_select(0, ids))
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        weight = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        weight.mul_(self._fast_row_scale_cached)
        return weight

    def reconstruct_weight_fast_norm(self, out_dtype: torch.dtype) -> Tensor:
        """Fast pre-row-scale reconstruct matching ``reconstruct_weight_fast`` internals."""
        if not self.supports_fast_reconstruct():
            return self.reconstruct_weight_norm(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        first = active_idx[0]
        ids0 = getattr(self, f"stage_ids_{first}").reshape(-1).to(torch.int64)
        recon = cb_list[first].index_select(0, ids0)
        for idx in active_idx[1:]:
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            recon.add_(cb_list[idx].index_select(0, ids))
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        return recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]

    def reconstruct_weight_hot(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        """Experimental M23 path: stage-local hot/cold codebook reconstruction.

        Each sub-stage keeps a small hot codebook of top ids chosen from the
        current encoding. Stages whose hot ids do not cover enough score mass
        fall back to the normal fast path.
        """
        if not self.supports_fast_reconstruct() or int(hot_topk) <= 0:
            return self.reconstruct_weight_fast(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_id_to_slot_cached is None
            or self._hot_positions_cached is None
            or self._hot_slots_cached is None
            or self._cold_positions_cached is None
            or self._cold_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        if all(item is None for item in self._hot_codebooks_cached):
            return self.reconstruct_weight_fast(out_dtype)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        recon = None
        for idx in active_idx:
            hot_codebook = self._hot_codebooks_cached[idx]
            hot_pos = self._hot_positions_cached[idx]
            hot_slots = self._hot_slots_cached[idx]
            cold_pos = self._cold_positions_cached[idx]
            cold_ids = self._cold_ids_cached[idx]
            if hot_codebook is None or hot_pos is None or hot_slots is None or cold_pos is None or cold_ids is None:
                ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
                stage = cb_list[idx].index_select(0, ids)
            else:
                stage_size = int(getattr(self, f"stage_ids_{idx}").numel())
                if hot_slots.numel() == stage_size:
                    stage = hot_codebook.index_select(0, hot_slots)
                elif hot_slots.numel() == 0:
                    stage = cb_list[idx].index_select(0, cold_ids)
                else:
                    stage = torch.empty((stage_size, self.block_size), dtype=out_dtype, device=device)
                    stage.index_copy_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    stage.index_copy_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
            if recon is None:
                recon = stage
            else:
                recon.add_(stage)
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        weight = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        weight.mul_(self._fast_row_scale_cached)
        return weight

    def _ensure_hot_recon_buffer(
        self, stage_size: int, out_dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """M26 stage 1: cached per-group recon buffer for hot/cold v2 path."""
        buf = self._hot_recon_buffer_cached
        if (
            buf is None
            or buf.shape[0] != stage_size
            or buf.shape[1] != self.block_size
            or buf.dtype != out_dtype
            or buf.device != device
        ):
            buf = torch.empty((stage_size, self.block_size), dtype=out_dtype, device=device)
            self._hot_recon_buffer_cached = buf
            self._hot_recon_buffer_dtype = out_dtype
            self._hot_recon_buffer_device = device
        return buf

    def reconstruct_weight_hot_v2(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        """M26 stage 1: allocation-free variant of reconstruct_weight_hot.

        Reuses a cached (stage_size, block_size) recon buffer per group. The
        first active sub-stage initialises the buffer via index_copy_; every
        subsequent active sub-stage accumulates in-place via index_add_.
        Numerically equivalent to ``reconstruct_weight_hot`` on the same
        encoding (and to ``reconstruct_weight_fast`` because hot/cold is exact),
        but eliminates the per-stage ``torch.empty`` allocation.
        """
        if not self.supports_fast_reconstruct() or int(hot_topk) <= 0:
            return self.reconstruct_weight_fast(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_id_to_slot_cached is None
            or self._hot_positions_cached is None
            or self._hot_slots_cached is None
            or self._cold_positions_cached is None
            or self._cold_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        if all(item is None for item in self._hot_codebooks_cached):
            return self.reconstruct_weight_fast(out_dtype)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        stage_size = int(rows) * int(blocks_per_row)
        recon = self._ensure_hot_recon_buffer(stage_size, out_dtype, device)
        first_done = False
        for idx in active_idx:
            hot_codebook = self._hot_codebooks_cached[idx]
            hot_pos = self._hot_positions_cached[idx]
            hot_slots = self._hot_slots_cached[idx]
            cold_pos = self._cold_positions_cached[idx]
            cold_ids = self._cold_ids_cached[idx]
            stage_fallback = (
                hot_codebook is None or hot_pos is None or hot_slots is None
                or cold_pos is None or cold_ids is None
            )
            if stage_fallback:
                ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
                stage = cb_list[idx].index_select(0, ids)
                if not first_done:
                    recon.copy_(stage)
                    first_done = True
                else:
                    recon.add_(stage)
                continue
            if not first_done:
                if hot_slots.numel() == stage_size:
                    recon.copy_(hot_codebook.index_select(0, hot_slots))
                elif hot_slots.numel() == 0:
                    recon.copy_(cb_list[idx].index_select(0, cold_ids))
                else:
                    recon.index_copy_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    recon.index_copy_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
                first_done = True
            else:
                if hot_slots.numel() == stage_size:
                    recon.index_add_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                elif hot_slots.numel() == 0:
                    recon.index_add_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
                else:
                    recon.index_add_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    recon.index_add_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
        weight_view = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        # NOTE: do not mul_ in place; recon is a cached buffer reused across forwards.
        return weight_view * self._fast_row_scale_cached

    def reconstruct_weight_norm(self, out_dtype: torch.dtype) -> Tensor:
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        flat_blocks = int(rows) * int(blocks_per_row)
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            from .block_vq import _sign_correction_matrix, _unpack_sign_bits

            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device).to(torch.float32)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            from .block_vq import _inverse_polar_transform

            recon = _inverse_polar_transform(recon)
        else:
            transform_matrix = self.transform_matrix
            if transform_matrix is None and self.transform_kind != "none":
                from .block_vq import _transform_matrix

                transform_matrix = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform_matrix is not None:
                recon = recon @ transform_matrix.to(torch.float32)
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        weight = recon.view(rows, blocks_per_row * self.block_size)[:, : self.in_features]
        return weight.to(out_dtype)

    def supports_stagewise_block_matmul(self) -> bool:
        return (
            self.block_scale is None
            and self.transform_kind == "none"
            and self.transform_matrix is None
            and self.transform_bias is None
            and self.residual_correction == "none"
            and self.residual_signs is None
            and self.residual_scale is None
        )

    def supports_triton_block_matmul(self) -> bool:
        if not self.supports_stagewise_block_matmul():
            return False
        if self.block_size != 32 or self.padded_cols != self.in_features:
            return False
        return all(getattr(self, f"stage_ids_{idx}").dtype == torch.uint8 for idx in range(self.num_stages))

    def _triton_state(self) -> tuple[Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("triton block matmul only supports plain uint8, block_size=32 encodings")
        device = self.row_scale.device
        if (
            self._triton_stage_ids is None
            or self._triton_codebooks is None
            or self._triton_cache_device != device
        ):
            self._triton_stage_ids = torch.stack(
                [getattr(self, f"stage_ids_{idx}").contiguous() for idx in range(self.num_stages)],
                dim=0,
            ).contiguous()
            codebooks = []
            for idx in range(self.num_stages):
                codebook = getattr(self, f"codebook_{idx}").contiguous()
                if self.stage_scales is not None:
                    codebook = (codebook.to(torch.float32) * self.stage_scales[idx].to(torch.float32)).to(codebook.dtype)
                codebooks.append(codebook)
            self._triton_codebooks = torch.stack(codebooks, dim=0).contiguous()
            self._triton_cache_device = device
        return self._triton_stage_ids, self._triton_codebooks

    def stagewise_block_matmul(self, x_flat: Tensor) -> Tensor:
        if not self.supports_stagewise_block_matmul():
            raise ValueError("stagewise_block_matmul only supports unnormalized, transform-free encodings")
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        if self.padded_cols == self.in_features:
            x_blocks = x_flat.reshape(-1, blocks_per_row, self.block_size)
        else:
            x_padded = F.pad(x_flat, (0, self.padded_cols - self.in_features), value=0.0)
            x_blocks = x_padded.reshape(-1, blocks_per_row, self.block_size)
        out = torch.zeros((x_blocks.shape[0], rows), device=x_blocks.device, dtype=x_blocks.dtype)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(x_blocks.dtype)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage_blocks = codebook[ids.reshape(-1).long()].view(rows, blocks_per_row, self.block_size).to(x_blocks.dtype)
            if stage_scales is not None:
                stage_blocks = stage_blocks * stage_scales[idx]
            out = out + torch.einsum("mbd,rbd->mr", x_blocks, stage_blocks)
        return out * self.row_scale.reshape(1, rows).to(out.dtype)

    def triton_block_matmul(self, x_flat: Tensor) -> Tensor:
        if not x_flat.is_cuda:
            raise ValueError("triton block matmul expects CUDA input")
        stage_ids, codebooks = self._triton_state()
        return rvq_group_linear_matmul(x_flat, stage_ids, codebooks, self.row_scale)

    def _triton_stage_local_hot_cold_state(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("stage_local_hot_cold requires plain uint8 block RVQ encodings")
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_id_to_slot_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        active_idx = self._compute_active_stage_indices()
        if (
            self._triton_stage_local_hot_ids is None
            or self._triton_stage_local_full_codebooks is None
            or self._triton_stage_local_hot_codebooks is None
            or self._triton_stage_local_hot_lut is None
            or self._triton_stage_local_cache_device != device
            or self._triton_stage_local_cache_dtype != out_dtype
            or self._triton_stage_local_cache_topk != int(hot_topk)
            or self._triton_stage_local_cache_score_mode != score_mode
            or self._triton_stage_local_cache_min_stage_share != float(min_stage_share)
            or self._triton_stage_local_cache_score_threshold_ratio != float(score_threshold_ratio)
            or int(self._triton_stage_local_hot_ids.shape[0]) != len(active_idx)
        ):
            stage_ids = []
            full_codebooks = []
            hot_codebooks = []
            hot_luts = []
            for idx in active_idx:
                full_cb = self._fast_codebooks_cached[idx].contiguous()
                hot_cb = self._hot_codebooks_cached[idx]
                hot_lut = self._hot_id_to_slot_cached[idx]
                stage_ids.append(getattr(self, f"stage_ids_{idx}").contiguous())
                full_codebooks.append(full_cb)
                if hot_cb is None or hot_lut is None:
                    hot_codebooks.append(torch.zeros((int(hot_topk), self.block_size), dtype=out_dtype, device=device))
                    hot_luts.append(torch.full((int(full_cb.shape[0]),), -1, dtype=torch.int16, device=device))
                    continue
                if int(hot_cb.shape[0]) < int(hot_topk):
                    pad = torch.zeros((int(hot_topk) - int(hot_cb.shape[0]), self.block_size), dtype=hot_cb.dtype, device=device)
                    hot_cb = torch.cat([hot_cb, pad], dim=0)
                hot_codebooks.append(hot_cb.contiguous())
                hot_luts.append(hot_lut.contiguous())
            self._triton_stage_local_hot_ids = torch.stack(stage_ids, dim=0).contiguous()
            self._triton_stage_local_full_codebooks = torch.stack(full_codebooks, dim=0).contiguous()
            self._triton_stage_local_hot_codebooks = torch.stack(hot_codebooks, dim=0).contiguous()
            self._triton_stage_local_hot_lut = torch.stack(hot_luts, dim=0).contiguous()
            self._triton_stage_local_cache_device = device
            self._triton_stage_local_cache_dtype = out_dtype
            self._triton_stage_local_cache_topk = int(hot_topk)
            self._triton_stage_local_cache_score_mode = score_mode
            self._triton_stage_local_cache_min_stage_share = float(min_stage_share)
            self._triton_stage_local_cache_score_threshold_ratio = float(score_threshold_ratio)
        return (
            self._triton_stage_local_hot_ids,
            self._triton_stage_local_full_codebooks,
            self._triton_stage_local_hot_codebooks,
            self._triton_stage_local_hot_lut,
            self._fast_row_scale_cached,
        )

    def triton_stage_local_hot_cold_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale = self._triton_stage_local_hot_cold_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_cold_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_codebooks,
            hot_lut,
            row_scale,
        )

    def _triton_stage_local_hot_palette_state(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("stage_local_hot_cold_b1 requires plain uint8 block RVQ encodings")
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_top_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        active_idx = self._compute_active_stage_indices()
        if (
            self._triton_stage_local_b1_stage_ids is None
            or self._triton_stage_local_b1_full_codebooks is None
            or self._triton_stage_local_b1_hot_ids is None
            or self._triton_stage_local_b1_hot_codebooks is None
            or self._triton_stage_local_b1_row_scale is None
            or self._triton_stage_local_b1_cache_device != device
            or self._triton_stage_local_b1_cache_dtype != out_dtype
            or self._triton_stage_local_b1_cache_topk != int(hot_topk)
            or self._triton_stage_local_b1_cache_score_mode != score_mode
            or self._triton_stage_local_b1_cache_min_stage_share != float(min_stage_share)
            or self._triton_stage_local_b1_cache_score_threshold_ratio != float(score_threshold_ratio)
            or int(self._triton_stage_local_b1_stage_ids.shape[0]) != len(active_idx)
        ):
            stage_ids = []
            full_codebooks = []
            hot_ids = []
            hot_codebooks = []
            for idx in active_idx:
                full_cb = self._fast_codebooks_cached[idx].contiguous()
                stage_ids.append(getattr(self, f"stage_ids_{idx}").contiguous())
                full_codebooks.append(full_cb)
                stage_hot_ids = None if self._hot_top_ids_cached is None else self._hot_top_ids_cached[idx]
                stage_hot_cb = self._hot_codebooks_cached[idx]
                if stage_hot_ids is None or stage_hot_cb is None:
                    hot_ids.append(torch.full((int(hot_topk),), -1, dtype=torch.int32, device=device))
                    hot_codebooks.append(torch.zeros((int(hot_topk), self.block_size), dtype=out_dtype, device=device))
                    continue
                hot_ids_i = stage_hot_ids[: int(hot_topk)].to(torch.int32).contiguous()
                hot_cb_i = stage_hot_cb[: int(hot_topk)].contiguous()
                if int(hot_ids_i.numel()) < int(hot_topk):
                    pad_ids = torch.full((int(hot_topk) - int(hot_ids_i.numel()),), -1, dtype=torch.int32, device=device)
                    pad_cb = torch.zeros((int(hot_topk) - int(hot_cb_i.shape[0]), self.block_size), dtype=out_dtype, device=device)
                    hot_ids_i = torch.cat([hot_ids_i, pad_ids], dim=0)
                    hot_cb_i = torch.cat([hot_cb_i, pad_cb], dim=0)
                hot_ids.append(hot_ids_i)
                hot_codebooks.append(hot_cb_i)
            self._triton_stage_local_b1_stage_ids = torch.stack(stage_ids, dim=0).contiguous()
            self._triton_stage_local_b1_full_codebooks = torch.stack(full_codebooks, dim=0).contiguous()
            self._triton_stage_local_b1_hot_ids = torch.stack(hot_ids, dim=0).contiguous()
            self._triton_stage_local_b1_hot_codebooks = torch.stack(hot_codebooks, dim=0).contiguous()
            self._triton_stage_local_b1_row_scale = self._fast_row_scale_cached
            self._triton_stage_local_b1_cache_device = device
            self._triton_stage_local_b1_cache_dtype = out_dtype
            self._triton_stage_local_b1_cache_topk = int(hot_topk)
            self._triton_stage_local_b1_cache_score_mode = score_mode
            self._triton_stage_local_b1_cache_min_stage_share = float(min_stage_share)
            self._triton_stage_local_b1_cache_score_threshold_ratio = float(score_threshold_ratio)
        return (
            self._triton_stage_local_b1_stage_ids,
            self._triton_stage_local_b1_full_codebooks,
            self._triton_stage_local_b1_hot_ids,
            self._triton_stage_local_b1_hot_codebooks,
            self._triton_stage_local_b1_row_scale,
        )

    def triton_stage_local_hot_palette_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_ids, hot_codebooks, row_scale = self._triton_stage_local_hot_palette_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_palette_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_ids,
            hot_codebooks,
            row_scale,
        )

    def triton_stage_local_hot_palette_b2_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale = self._triton_stage_local_hot_cold_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_palette_b2_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_codebooks,
            hot_lut,
            row_scale,
        )


class PackedGroupedBlockRVQLinear(nn.Module):
    def __init__(
        self,
        enc: GroupedBlockRVQEncoding,
        bias: Tensor | None = None,
        *,
        matmul_strategy: str = "per_group",
        matmul_chunk_rows: int | None = None,
        local_palette_group_cols: int | None = None,
        hot_topk: int | None = None,
        hot_score_mode: str = "row_scale_norm",
        hot_min_stage_share: float = 0.0,
        hot_score_threshold_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if matmul_strategy == "triton_hot_cold_persistent" and local_palette_group_cols is None:
            local_palette_group_cols = 256
        if matmul_strategy not in {"per_group", "per_group_fast", "full_weight", "full_weight_fast", "full_weight_hot", "full_weight_hot_v2", "chunked_weight", "local_palette", "triton_hot_cold_persistent", "stage_local_hot_cold", "stage_local_hot_cold_b1", "stage_local_hot_cold_b2", "stage_local_hot_cold_b3", "stagewise_einsum", "triton_block_rvq", "stacked_matmul"}:
            raise ValueError(f"unsupported matmul_strategy: {matmul_strategy}")
        if matmul_strategy == "chunked_weight" and (matmul_chunk_rows is None or int(matmul_chunk_rows) <= 0):
            raise ValueError("chunked_weight requires a positive matmul_chunk_rows")
        if matmul_strategy in {"local_palette", "triton_hot_cold_persistent"} and (local_palette_group_cols is None or int(local_palette_group_cols) <= 0):
            raise ValueError(f"{matmul_strategy} requires a positive local_palette_group_cols")
        if matmul_strategy == "full_weight_hot" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("full_weight_hot requires a positive hot_topk")
        if matmul_strategy == "full_weight_hot_v2" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("full_weight_hot_v2 requires a positive hot_topk")
        if matmul_strategy == "triton_hot_cold_persistent" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("triton_hot_cold_persistent requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b1" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b1 requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b2" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b2 requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b3" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b3 requires a positive hot_topk")
        self.in_features = int(enc.original_shape[1])
        self.out_features = int(enc.original_shape[0])
        self.groups = nn.ModuleList([PackedBlockRVQGroup(group) for group in enc.groups])
        self.row_slices = tuple(enc.row_slices)
        self.matmul_strategy = matmul_strategy
        self.matmul_chunk_rows = None if matmul_chunk_rows is None else int(matmul_chunk_rows)
        self.local_palette_group_cols = None if local_palette_group_cols is None else int(local_palette_group_cols)
        self.hot_topk = None if hot_topk is None else int(hot_topk)
        self.hot_score_mode = hot_score_mode
        self.hot_min_stage_share = float(hot_min_stage_share)
        self.hot_score_threshold_ratio = float(hot_score_threshold_ratio)
        self._local_palette_plan: list[dict[str, Tensor | int]] | None = None
        self._local_palette_plan_device: torch.device | None = None
        self._local_palette_row_scale: Tensor | None = None
        self._local_hotprefix_plan: list[dict[str, Tensor | int]] | None = None
        self._local_hotprefix_plan_device: torch.device | None = None
        self._local_hotprefix_row_scale: Tensor | None = None
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16).contiguous())
        else:
            self.bias = None

    def reconstruct_weight(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [group.reconstruct_weight(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_fast(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [group.reconstruct_weight_fast(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_hot(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [
            group.reconstruct_weight_hot(
                out_dtype=out_dtype,
                hot_topk=int(self.hot_topk),
                score_mode=self.hot_score_mode,
                min_stage_share=self.hot_min_stage_share,
                score_threshold_ratio=self.hot_score_threshold_ratio,
            )
            for group in self.groups
        ]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_hot_v2(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [
            group.reconstruct_weight_hot_v2(
                out_dtype=out_dtype,
                hot_topk=int(self.hot_topk),
                score_mode=self.hot_score_mode,
                min_stage_share=self.hot_min_stage_share,
                score_threshold_ratio=self.hot_score_threshold_ratio,
            )
            for group in self.groups
        ]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_norm(self, out_dtype: torch.dtype = torch.float16) -> Tensor:
        parts = [group.reconstruct_weight_norm(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def _local_palette_group_rows(self) -> int:
        return max((row1 - row0) for row0, row1 in self.row_slices)

    def _local_palette_state(self) -> tuple[list[dict[str, Tensor | int]], Tensor]:
        device = self.groups[0].row_scale.device
        if self._local_palette_plan is None or self._local_palette_plan_device != device:
            routed_norm = self.reconstruct_weight_norm(out_dtype=torch.float16)
            self._local_palette_plan = build_grouped_local_plan(
                routed_norm,
                self._local_palette_group_rows(),
                int(self.local_palette_group_cols),
            )
            self._local_palette_row_scale = torch.cat([group.row_scale for group in self.groups], dim=0).contiguous()
            self._local_palette_plan_device = device
        return self._local_palette_plan, self._local_palette_row_scale

    def _local_hotprefix_state(self) -> tuple[list[dict[str, Tensor | int]], Tensor]:
        device = self.groups[0].row_scale.device
        if self._local_hotprefix_plan is None or self._local_hotprefix_plan_device != device:
            routed_norm = torch.cat(
                [group.reconstruct_weight_fast_norm(out_dtype=torch.bfloat16) for group in self.groups],
                dim=0,
            )
            plan_cpu = build_grouped_local_hotprefix_plan(
                routed_norm.cpu(),
                self._local_palette_group_rows(),
                int(self.local_palette_group_cols),
            )
            self._local_hotprefix_plan = [
                {
                    **item,
                    "palette": item["palette"].to(device),
                    "local_idx": item["local_idx"].to(device),
                }
                for item in plan_cpu
            ]
            self._local_hotprefix_row_scale = torch.cat([group.row_scale for group in self.groups], dim=0).to(torch.bfloat16).contiguous()
            self._local_hotprefix_plan_device = device
        return self._local_hotprefix_plan, self._local_hotprefix_row_scale

    def _chunked_linear(self, x_flat: Tensor) -> Tensor:
        outputs = []
        weight_parts = []
        bias_parts = []
        rows_accum = 0
        for (row0, row1), group in zip(self.row_slices, self.groups):
            weight_parts.append(group.reconstruct_weight(out_dtype=x_flat.dtype))
            if self.bias is not None:
                bias_parts.append(self.bias[row0:row1].to(x_flat.dtype))
            rows_accum += row1 - row0
            if rows_accum >= int(self.matmul_chunk_rows):
                weight = torch.cat(weight_parts, dim=0)
                bias = None if self.bias is None else torch.cat(bias_parts, dim=0)
                outputs.append(F.linear(x_flat, weight, bias))
                weight_parts.clear()
                bias_parts.clear()
                rows_accum = 0
        if weight_parts:
            weight = torch.cat(weight_parts, dim=0)
            bias = None if self.bias is None else torch.cat(bias_parts, dim=0)
            outputs.append(F.linear(x_flat, weight, bias))
        return torch.cat(outputs, dim=-1)

    def _stacked_group_linear(self, x_flat: Tensor) -> Tensor:
        out = torch.empty((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)
        buckets: dict[int, list[tuple[int, int, PackedBlockRVQGroup]]] = {}
        for (row0, row1), group in zip(self.row_slices, self.groups):
            buckets.setdefault(row1 - row0, []).append((row0, row1, group))
        x_batch = x_flat.unsqueeze(0)
        for rows, items in buckets.items():
            if len(items) == 1:
                row0, row1, group = items[0]
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                out[:, row0:row1] = F.linear(x_flat, group.reconstruct_weight(out_dtype=x_flat.dtype), bias)
                continue
            weights = torch.stack(
                [group.reconstruct_weight(out_dtype=x_flat.dtype) for _, _, group in items],
                dim=0,
            )
            bucket_out = torch.matmul(x_batch, weights.transpose(1, 2).contiguous())
            if self.bias is not None:
                bias = torch.stack(
                    [self.bias[row0:row1].to(x_flat.dtype) for row0, row1, _ in items],
                    dim=0,
                )
                bucket_out = bucket_out + bias.unsqueeze(1)
            for idx, (row0, row1, _) in enumerate(items):
                out[:, row0:row1] = bucket_out[idx]
        return out

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        if self.matmul_strategy == "full_weight":
            # Avoid dozens of tiny GEMM launches when attention-side row groups are small.
            weight = self.reconstruct_weight(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_fast":
            # M20: same as full_weight but uses cached fast reconstruct per group.
            weight = self.reconstruct_weight_fast(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_hot":
            weight = self.reconstruct_weight_hot(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_hot_v2":
            weight = self.reconstruct_weight_hot_v2(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "chunked_weight":
            out = self._chunked_linear(x_flat)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "local_palette":
            plan, row_scale = self._local_palette_state()
            out = full_layer_grouped_local_matmul(x_flat, plan, row_scale)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "triton_hot_cold_persistent":
            plan, row_scale = self._local_hotprefix_state()
            out = full_layer_grouped_local_hotprefix_matmul(x_flat, plan, row_scale, hot_size=int(self.hot_topk))
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_cold_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b1":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b2":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_b2_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b3":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_b2_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=0.0,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stagewise_einsum":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.stagewise_block_matmul(x_flat)
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "triton_block_rvq":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                if x_flat.is_cuda and group.supports_triton_block_matmul():
                    part = group.triton_block_matmul(x_flat)
                else:
                    part = F.linear(x_flat, group.reconstruct_weight(out_dtype=x_flat.dtype), bias=None)
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stacked_matmul":
            out = self._stacked_group_linear(x_flat)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "per_group_fast":
            # M20: per-group with cached bf16 codebooks and reusable per-group reconstruct buffers.
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                weight = group.reconstruct_weight_fast(out_dtype=x_flat.dtype)
                parts.append(F.linear(x_flat, weight, bias))
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        parts = []
        for (row0, row1), group in zip(self.row_slices, self.groups):
            bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
            weight = group.reconstruct_weight(out_dtype=x_flat.dtype)
            parts.append(F.linear(x_flat, weight, bias))
        out = torch.cat(parts, dim=-1)
        return out.reshape(*orig_shape[:-1], self.out_features)


class EagerFp8Linear(nn.Module):
    """Materialize route weight to FP8 (e4m3) with per-row scale, run via torch._scaled_mm.

    H200-native FP8 storage path: ~50% VRAM versus ``EagerBf16Linear``.

    Numerical recipe (best-quality FP8 inference combo on H200):
      * Weight: e4m3 (more precision, less range) with per-output-row scale.
      * Activation: e5m2 (more range, less precision) with per-token scale.
      * Output: bf16 to preserve residual stream precision.

    The route row_scale already captures per-row dynamic range optimally; we
    re-derive it here from the decoded amax to honor the actual fp8 codomain.
    """

    FP8_W_MAX = 448.0    # max |x| representable in float8_e4m3fn
    FP8_A_MAX = 57344.0  # max |x| representable in float8_e5m2

    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        x_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__()
        out_features = int(ids.shape[0])
        in_features = int(ids.shape[1])
        self.in_features = in_features
        self.out_features = out_features
        self.x_dtype = x_dtype
        self._x_max = self.FP8_A_MAX if x_dtype is torch.float8_e5m2 else self.FP8_W_MAX

        codebook_sum = codebook_w.sum(dim=-1).to(torch.float32)
        weight_fp32 = codebook_sum[ids.long()] * row_scale.to(torch.float32)
        amax = weight_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        weight_scale = (amax / self.FP8_W_MAX).to(torch.float32)
        w_q = (weight_fp32 / weight_scale).clamp(-self.FP8_W_MAX, self.FP8_W_MAX).to(torch.float8_e4m3fn)
        del weight_fp32

        self.register_buffer("weight_fp8", w_q.contiguous())
        self.register_buffer("weight_scale", weight_scale.view(1, -1).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16).contiguous())
        else:
            self.bias = None

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        x_dtype: torch.dtype = torch.float8_e4m3fn,
        **_: object,
    ) -> "EagerFp8Linear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, x_dtype=x_dtype)

    def reconstruct_weight(self) -> Tensor:
        return (self.weight_fp8.float() * self.weight_scale.view(-1, 1).float())

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        x_scale = (x_amax.float() / self._x_max)
        x_q = (x_flat / x_amax * self._x_max).clamp(-self._x_max, self._x_max).to(self.x_dtype)
        out = torch._scaled_mm(
            x_q,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)


class AdaptiveFusedIDRouteLinear(PackedIDRouteLinear):
    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        validate_calls: int = 1,
        disable_primary_after_nonfinite: bool = True,
        shadow_validate: bool = False,
        shadow_rel_mse_tol: float = 1e-4,
    ) -> None:
        super().__init__(ids, codebook_w, row_scale, bias)
        self.validate_calls = max(int(validate_calls), 0)
        self.disable_primary_after_nonfinite = disable_primary_after_nonfinite
        self.shadow_validate = shadow_validate
        self.shadow_rel_mse_tol = float(shadow_rel_mse_tol)
        self._validation_calls_left = self.validate_calls
        self._primary_enabled = True
        self.fallback_count = 0
        self.shadow_mismatch_count = 0
        self.last_validation_rel_mse: float | None = None

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        validate_calls: int = 1,
        disable_primary_after_nonfinite: bool = True,
        shadow_validate: bool = False,
        shadow_rel_mse_tol: float = 1e-4,
        **_: object,
    ) -> "AdaptiveFusedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(
            ids,
            codebook_w,
            row_scale.to(torch.float16),
            bias,
            validate_calls=validate_calls,
            disable_primary_after_nonfinite=disable_primary_after_nonfinite,
            shadow_validate=shadow_validate,
            shadow_rel_mse_tol=shadow_rel_mse_tol,
        )

    def _primary_forward(self, x: Tensor) -> Tensor:
        out = id_route_linear_matmul(x, self.ids, self.codebook_sum, self.row_scale)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if not self._primary_enabled:
            return super().forward(x)
        try:
            out = self._primary_forward(x)
        except RuntimeError:
            self.fallback_count += 1
            if self.disable_primary_after_nonfinite:
                self._primary_enabled = False
            return super().forward(x)
        if self._validation_calls_left > 0:
            self._validation_calls_left -= 1
            if not bool(torch.isfinite(out).all().item()):
                self.fallback_count += 1
                if self.disable_primary_after_nonfinite:
                    self._primary_enabled = False
                return super().forward(x)
            if self.shadow_validate:
                reference = super().forward(x)
                denom = reference.float().square().mean().clamp_min(1e-12)
                rel_mse = float(((out.float() - reference.float()).square().mean() / denom).item())
                self.last_validation_rel_mse = rel_mse
                if rel_mse > self.shadow_rel_mse_tol:
                    self.shadow_mismatch_count += 1
                    if self.disable_primary_after_nonfinite:
                        self._primary_enabled = False
                    return reference
        return out


@torch.no_grad()
def quantize_linear_to_packed(
    linear: nn.Linear,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    runtime_cls: type[nn.Module] = PackedIDRouteLinear,
    runtime_kwargs: dict[str, object] | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    weight = linear.weight.detach()
    row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = weight / row_scale
    sample = w_norm.flatten()
    if sample.numel() > sample_limit:
        pick = torch.randint(0, sample.numel(), (sample_limit,), device=sample.device)
        sample = sample[pick]
    ladder = calibrate_ladder(
        sample,
        l_max=l_max,
        refine_iters=20,
        pin_top=True,
        top_value=1.0,
        seed="geometric",
    )
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    codebook, ids = build_codebook(enc.digits, enc.stop_depth, l_max=l_max)
    packed = runtime_cls.from_encoded(ids, codebook.digits, ladder, row_scale, linear.bias, **(runtime_kwargs or {}))
    stats = {
        "rel_mse": float(rel_mse(weight.float(), packed.reconstruct_weight().float()).item()),
        "unique_routes": int(codebook.size),
        "id_bits": int(math.ceil(math.log2(max(codebook.size, 2)))),
    }
    return packed, stats


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _load_shape_runtime_policy(shape_policy_json: str | Path | None) -> dict[str, dict[str, int]]:
    if shape_policy_json is None:
        return {}
    with open(shape_policy_json) as handle:
        payload = json.load(handle)
    return {
        str(item["tensor_name"]): {
            "group_rows": int(item["group_rows"]),
            "group_cols": int(item["group_cols"]),
        }
        for item in payload.get("selected", [])
    }


def _resolve_shape_policy(
    shape_policy: dict[str, dict[str, int]],
    module_name: str,
) -> tuple[str, dict[str, int]] | None:
    candidates = [f"{module_name}.weight"]
    if module_name.startswith("model."):
        candidates.append(f"{module_name[6:]}.weight")
    else:
        candidates.append(f"model.{module_name}.weight")
    for tensor_name in candidates:
        match = shape_policy.get(tensor_name)
        if match is not None:
            return tensor_name, match
    return None


def _load_fused_allowlist(fused_allowlist_json: str | Path | None) -> set[str]:
    if fused_allowlist_json is None:
        return set()
    with open(fused_allowlist_json) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if "promoted_layers" in payload:
        return {str(item) for item in payload["promoted_layers"]}
    adaptive_names = payload.get("adaptive_names")
    if isinstance(adaptive_names, dict) and "primary_enabled_layer_names" in adaptive_names:
        return {str(item) for item in adaptive_names["primary_enabled_layer_names"]}
    return set()


def _module_name_candidates(module_name: str) -> tuple[str, ...]:
    if module_name.startswith("model."):
        return (module_name, module_name[6:])
    return (module_name, f"model.{module_name}")


@torch.no_grad()
def replace_packed_id_route_layers(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    runtime_cls: type[nn.Module] = PackedIDRouteLinear,
    runtime_kwargs: dict[str, object] | None = None,
    shape_policy_json: str | Path | None = None,
) -> list[dict[str, float]]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes)
    ]
    shape_policy = _load_shape_runtime_policy(shape_policy_json)
    stats = []
    default_runtime_kwargs = dict(runtime_kwargs or {})
    for name, module in targets:
        layer_runtime_kwargs: dict[str, object] | None = None
        selected = _resolve_shape_policy(shape_policy, name)
        if selected is None:
            layer_runtime_cls = runtime_cls
            layer_runtime_kwargs = dict(default_runtime_kwargs)
            tensor_name = f"{name}.weight"
            if runtime_cls is FusedIDRouteLinear:
                runtime_decision = "global_id_triton"
            elif runtime_cls is AdaptiveFusedIDRouteLinear:
                runtime_decision = "adaptive_global_id"
            elif runtime_cls is CachedPackedIDRouteLinear:
                runtime_decision = "cached_packed"
            elif runtime_cls is PackedIDRouteLinear:
                runtime_decision = "packed_materialize"
            elif runtime_cls is EagerBf16Linear:
                runtime_decision = "eager_bf16"
            elif runtime_cls is EagerFp8Linear:
                runtime_decision = "eager_fp8"
            else:
                runtime_decision = runtime_cls.__name__
        else:
            tensor_name, layer_runtime_kwargs = selected
            layer_runtime_cls = GroupedLocalRouteLinear
            runtime_decision = "local_palette_grouped"
        packed, layer_stats = quantize_linear_to_packed(
            module,
            l_max=l_max,
            sample_limit=sample_limit,
            runtime_cls=layer_runtime_cls,
            runtime_kwargs=layer_runtime_kwargs,
        )
        _set_submodule(model, name, packed)
        # Explicitly drop the original Linear's weight/bias so the bf16 buffers
        # are released back to the allocator before the next iteration. Without
        # this the loop variable keeps the old W-sized tensor alive and the peak
        # VRAM doubles (old nn.Linear.weight + new packed/eager weight).
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        item: dict[str, float | str | int] = {
            "name": name,
            "tensor_name": tensor_name,
            "runtime_decision": runtime_decision,
            **layer_stats,
        }
        if layer_runtime_kwargs is not None:
            item.update(
                {
                    key: value
                    for key, value in layer_runtime_kwargs.items()
                    if isinstance(value, (bool, int, float, str))
                }
            )
        stats.append(item)
    return stats


@torch.no_grad()
def replace_with_deployment_runtime(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    cache_max_mb: int = 128,
) -> list[dict[str, float]]:
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=CachedPackedIDRouteLinear,
        runtime_kwargs={"max_cache_bytes": max(int(cache_max_mb), 0) * 2**20},
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_bf16(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    """Route-decode weights once, store as plain bf16 buffers.

    Preserves the 3-bit disk/codebook PPL signature but matches the baseline
    nn.Linear runtime footprint exactly (speed + VRAM).
    """
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=EagerBf16Linear,
        runtime_kwargs={"target_dtype": target_dtype},
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_fp8(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
) -> list[dict[str, float]]:
    """Route-decode weights then store as fp8_e4m3 with per-row scale (~50% of bf16 VRAM).

    Use when VRAM is the bottleneck. Compute uses ``torch._scaled_mm`` with
    rowwise scaling and bf16 output. Per-token x scaling is dynamic.
    """
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=EagerFp8Linear,
        runtime_kwargs=None,
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_hybrid(
    model: nn.Module,
    fp8_suffixes: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "mlp.gate_proj", "mlp.up_proj",
    ),
    bf16_suffixes: tuple[str, ...] = ("self_attn.o_proj", "mlp.down_proj"),
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    """Hybrid eager runtime: FP8 for attention QKV + MLP gate/up, BF16 for residual writebacks.

    Rationale: ``o_proj`` and ``down_proj`` write directly into the residual
    stream where small numerical errors compound across layers. Keep them in
    bf16 for quality. The other five projections (Q, K, V, gate, up) feed
    nonlinearities (softmax, swiglu) that absorb fp8 noise.
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(bf16_suffixes):
            targets.append((name, module, "bf16"))
        elif name.endswith(fp8_suffixes):
            targets.append((name, module, "fp8"))
    stats = []
    for name, module, kind in targets:
        if kind == "fp8":
            cls = EagerFp8Linear
            kwargs: dict[str, object] | None = None
            decision = "eager_fp8"
        else:
            cls = EagerBf16Linear
            kwargs = {"target_dtype": target_dtype}
            decision = "eager_bf16"
        packed, layer_stats = quantize_linear_to_packed(
            module, l_max=l_max, sample_limit=sample_limit,
            runtime_cls=cls, runtime_kwargs=kwargs,
        )
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append({
            "name": name,
            "tensor_name": f"{name}.weight",
            "runtime_decision": decision,
            **layer_stats,
        })
    return stats


@torch.no_grad()
def replace_with_eager_block_rvq(
    model: nn.Module,
    target_module_names: tuple[str, ...],
    *,
    group_rows: int = 2048,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 4,
    product_splits: int = 1,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 65_536,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    target_set = set(target_module_names)
    stats = []
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encode_grouped_block_residual_vq(
            original_weight,
            group_rows=group_rows,
            block_size=block_size,
            codebook_size=codebook_size,
            num_stages=num_stages,
            product_splits=product_splits,
            normalize_blocks=normalize_blocks,
            transform_kind=transform_kind,
            calibrate_stage_scales=calibrate_stage_scales,
            residual_correction=residual_correction,
            sample_limit=sample_limit,
            kmeans_iters=kmeans_iters,
            batch_size=batch_size,
        )
        approx_weight = enc.reconstruct(out_dtype=target_dtype)
        packed = EagerBlockRVQLinear(approx_weight, module.bias, target_dtype=target_dtype)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "eager_block_rvq",
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
                "group_rows": int(group_rows),
                "block_size": int(block_size),
                "codebook_size": int(codebook_size),
                "num_stages": int(num_stages),
                "product_splits": int(product_splits),
                "normalize_blocks": normalize_blocks,
                "transform_kind": transform_kind,
                "calibrate_stage_scales": bool(calibrate_stage_scales),
                "residual_correction": residual_correction,
            }
        )
    return stats


@torch.no_grad()
def replace_with_packed_block_rvq(
    model: nn.Module,
    target_module_names: tuple[str, ...],
    *,
    group_rows: int = 2048,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 4,
    product_splits: int = 1,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 65_536,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
    matmul_strategy: str = "per_group",
    matmul_chunk_rows: int | None = None,
    local_palette_group_cols: int | None = None,
    hot_topk: int | None = None,
    hot_score_mode: str = "row_scale_norm",
    hot_min_stage_share: float = 0.0,
    hot_score_threshold_ratio: float = 0.0,
) -> list[dict[str, float]]:
    target_set = set(target_module_names)
    stats = []
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encode_grouped_block_residual_vq(
            original_weight,
            group_rows=group_rows,
            block_size=block_size,
            codebook_size=codebook_size,
            num_stages=num_stages,
            product_splits=product_splits,
            normalize_blocks=normalize_blocks,
            transform_kind=transform_kind,
            calibrate_stage_scales=calibrate_stage_scales,
            residual_correction=residual_correction,
            sample_limit=sample_limit,
            kmeans_iters=kmeans_iters,
            batch_size=batch_size,
        )
        approx_weight = enc.reconstruct(out_dtype=torch.bfloat16)
        packed = PackedGroupedBlockRVQLinear(
            enc,
            module.bias,
            matmul_strategy=matmul_strategy,
            matmul_chunk_rows=matmul_chunk_rows,
            local_palette_group_cols=local_palette_group_cols,
            hot_topk=hot_topk,
            hot_score_mode=hot_score_mode,
            hot_min_stage_share=hot_min_stage_share,
            hot_score_threshold_ratio=hot_score_threshold_ratio,
        ).to(original_weight.device)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "packed_block_rvq",
                "matmul_strategy": matmul_strategy,
                "matmul_chunk_rows": int(matmul_chunk_rows) if matmul_chunk_rows is not None else None,
                "local_palette_group_cols": int(local_palette_group_cols) if local_palette_group_cols is not None else None,
                "hot_topk": int(hot_topk) if hot_topk is not None else None,
                "hot_score_mode": hot_score_mode,
                "hot_min_stage_share": float(hot_min_stage_share),
                "hot_score_threshold_ratio": float(hot_score_threshold_ratio),
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
                "group_rows": int(group_rows),
                "block_size": int(block_size),
                "codebook_size": int(codebook_size),
                "num_stages": int(num_stages),
                "product_splits": int(product_splits),
                "normalize_blocks": normalize_blocks,
                "transform_kind": transform_kind,
                "calibrate_stage_scales": bool(calibrate_stage_scales),
                "residual_correction": residual_correction,
            }
        )
    return stats


@torch.no_grad()
def replace_with_preencoded_packed_block_rvq(
    model: nn.Module,
    encodings_by_name: dict[str, GroupedBlockRVQEncoding],
    *,
    matmul_strategy: str = "per_group",
    matmul_chunk_rows: int | None = None,
    local_palette_group_cols: int | None = None,
    hot_topk: int | None = None,
    hot_score_mode: str = "row_scale_norm",
    hot_min_stage_share: float = 0.0,
    hot_score_threshold_ratio: float = 0.0,
) -> list[dict[str, float]]:
    stats = []
    for name, module in model.named_modules():
        if name not in encodings_by_name or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encodings_by_name[name]
        approx_weight = enc.reconstruct(out_dtype=torch.bfloat16).to(original_weight.device)
        packed = PackedGroupedBlockRVQLinear(
            enc,
            module.bias,
            matmul_strategy=matmul_strategy,
            matmul_chunk_rows=matmul_chunk_rows,
            local_palette_group_cols=local_palette_group_cols,
            hot_topk=hot_topk,
            hot_score_mode=hot_score_mode,
            hot_min_stage_share=hot_min_stage_share,
            hot_score_threshold_ratio=hot_score_threshold_ratio,
        ).to(original_weight.device)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "preencoded_packed_block_rvq",
                "matmul_strategy": matmul_strategy,
                "matmul_chunk_rows": int(matmul_chunk_rows) if matmul_chunk_rows is not None else None,
                "local_palette_group_cols": int(local_palette_group_cols) if local_palette_group_cols is not None else None,
                "hot_topk": int(hot_topk) if hot_topk is not None else None,
                "hot_score_mode": hot_score_mode,
                "hot_min_stage_share": float(hot_min_stage_share),
                "hot_score_threshold_ratio": float(hot_score_threshold_ratio),
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
            }
        )
    return stats


@torch.no_grad()
def replace_with_hybrid_runtime(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    cache_max_mb: int = 128,
    fused_allowlist_json: str | Path | None = None,
    promoted_runtime_cls: type[nn.Module] = FusedIDRouteLinear,
    promoted_runtime_kwargs: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes)
    ]
    shape_policy = _load_shape_runtime_policy(shape_policy_json)
    fused_allowlist = _load_fused_allowlist(fused_allowlist_json)
    cached_runtime_kwargs = {"max_cache_bytes": max(int(cache_max_mb), 0) * 2**20}
    stats = []
    for name, module in targets:
        selected = _resolve_shape_policy(shape_policy, name)
        if selected is not None:
            tensor_name, layer_runtime_kwargs = selected
            layer_runtime_cls = GroupedLocalRouteLinear
            runtime_decision = "local_palette_grouped"
        elif any(candidate in fused_allowlist for candidate in _module_name_candidates(name)):
            tensor_name = f"{name}.weight"
            layer_runtime_kwargs = dict(promoted_runtime_kwargs or {})
            layer_runtime_cls = promoted_runtime_cls
            if promoted_runtime_cls is FusedIDRouteLinear:
                runtime_decision = "promoted_fused_global_id"
            elif promoted_runtime_cls is AdaptiveFusedIDRouteLinear:
                runtime_decision = "promoted_adaptive_global_id"
            else:
                runtime_decision = f"promoted_{promoted_runtime_cls.__name__}"
        else:
            tensor_name = f"{name}.weight"
            layer_runtime_kwargs = dict(cached_runtime_kwargs)
            layer_runtime_cls = CachedPackedIDRouteLinear
            runtime_decision = "cached_packed_default"
        packed, layer_stats = quantize_linear_to_packed(
            module,
            l_max=l_max,
            sample_limit=sample_limit,
            runtime_cls=layer_runtime_cls,
            runtime_kwargs=layer_runtime_kwargs,
        )
        _set_submodule(model, name, packed)
        item: dict[str, float | str | int] = {
            "name": name,
            "tensor_name": tensor_name,
            "runtime_decision": runtime_decision,
            **layer_stats,
        }
        item.update(
            {
                key: value
                for key, value in layer_runtime_kwargs.items()
                if isinstance(value, (bool, int, float, str))
            }
        )
        stats.append(item)
    return stats


# ---------------------------------------------------------------------------
# M21: global variable-stage decoding helpers
# ---------------------------------------------------------------------------

def set_global_effective_stages(model: nn.Module, stages: int) -> int:
    """Set ``effective_stages_per_split`` on every ``PackedBlockRVQGroup``.

    With product_splits=ps and stages_per_split=s, calling this with k means
    "keep the first k stages of each split", effectively running k*ps gather+add
    operations instead of s*ps. Returns the number of groups updated.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, PackedBlockRVQGroup):
            module.set_effective_stages_per_split(int(stages))
            n += 1
    return n


def set_effective_stages_by_name(model: nn.Module, attn_stages: int | None, mlp_stages: int | None) -> dict[str, int]:
    """M22: per-role stage cap. Routes stage cap by parent module name pattern.

    ``.self_attn.`` modules (q/k/v/o) get ``attn_stages``;
    ``.mlp.`` modules (gate/up/down) get ``mlp_stages``.
    Returns counts per bucket.
    """
    counts = {"attn": 0, "mlp": 0, "other": 0}
    for name, module in model.named_modules():
        if not isinstance(module, PackedGroupedBlockRVQLinear):
            continue
        bucket = "attn" if ".self_attn." in name else ("mlp" if ".mlp." in name else "other")
        target = attn_stages if bucket == "attn" else (mlp_stages if bucket == "mlp" else None)
        if target is None:
            continue
        for sub in module.modules():
            if isinstance(sub, PackedBlockRVQGroup):
                sub.set_effective_stages_per_split(int(target))
        counts[bucket] += 1
    return counts


def set_effective_stages_from_map(model: nn.Module, name_to_k: dict[str, int]) -> int:
    """M24: apply per-layer stage cap from a {layer_name: chosen_k} map.

    Returns count of layers actually updated. Names without matches are skipped.
    """
    updated = 0
    for name, module in model.named_modules():
        if not isinstance(module, PackedGroupedBlockRVQLinear):
            continue
        if name not in name_to_k:
            continue
        k = int(name_to_k[name])
        for sub in module.modules():
            if isinstance(sub, PackedBlockRVQGroup):
                sub.set_effective_stages_per_split(k)
        updated += 1
    return updated
