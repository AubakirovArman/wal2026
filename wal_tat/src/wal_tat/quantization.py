"""Group-wise ternary quantization primitives used by WAL-TAT."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def group_shape(in_features: int, group_size: int) -> Tuple[int, int, int]:
    """Return effective group size, right padding, and number of groups."""
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    size = in_features if group_size <= 0 or group_size >= in_features else int(group_size)
    padding = (-in_features) % size
    return size, padding, (in_features + padding) // size


def padded_grouped(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int, int]:
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    size, padding, groups = group_shape(weight.shape[1], group_size)
    value = weight.float()
    if padding:
        value = F.pad(value, (0, padding))
    return value.view(weight.shape[0], groups, size), padding, size


def initial_group_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped, _, _ = padded_grouped(weight.detach(), group_size)
    return grouped.abs().mean(-1).clamp_min(1e-5)


def hestia_quantize(
    weight: torch.Tensor,
    *,
    group_size: int,
    pressure: float,
    temperature: float,
    scales: Optional[torch.Tensor] = None,
    ste: bool = True,
    soft_chunk_rows: int = 256,
) -> torch.Tensor:
    """Soft-to-hard expectation over ``{-1, 0, +1}`` with an STE hard limit."""
    original_dtype = weight.dtype
    grouped, padding, _ = padded_grouped(weight, group_size)
    if scales is None:
        effective_scales = grouped.abs().mean(-1, keepdim=True).clamp_min(1e-5)
    else:
        effective_scales = scales.float().abs().clamp_min(1e-5).unsqueeze(-1)
        if effective_scales.shape[:2] != grouped.shape[:2]:
            raise ValueError("scales do not match grouped weight")
    normalized = grouped / effective_scales

    if temperature > 0:
        codebook = torch.tensor([-1.0, 0.0, 1.0], device=weight.device)
        rows = grouped.shape[0] if soft_chunk_rows <= 0 else soft_chunk_rows
        chunks = []
        for start in range(0, grouped.shape[0], rows):
            logits = -(normalized[start : start + rows].unsqueeze(-1) - codebook).square()
            probabilities = torch.softmax(logits / (temperature + 1e-6), dim=-1)
            expectation = (probabilities * codebook).sum(-1)
            chunks.append(expectation * effective_scales[start : start + rows])
        quantized = torch.cat(chunks, dim=0)
    else:
        codes = normalized.round().clamp(-1, 1)
        hard = codes * effective_scales
        if ste and scales is not None:
            quantized = codes.detach() * effective_scales + grouped - grouped.detach()
        else:
            quantized = grouped + (hard - grouped).detach() if ste else hard

    flat = quantized.reshape(weight.shape[0], -1)[:, : weight.shape[1]]
    pressure = min(max(float(pressure), 0.0), 1.0)
    result = torch.lerp(weight.float(), flat, pressure)
    return result.to(original_dtype)


@torch.no_grad()
def hard_codes_scales(
    weight: torch.Tensor,
    group_size: int,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grouped, _, _ = padded_grouped(weight.detach(), group_size)
    effective_scales = (
        grouped.abs().mean(-1).clamp_min(1e-5)
        if scales is None
        else scales.float().abs().clamp_min(1e-5)
    )
    codes = (grouped / effective_scales.unsqueeze(-1)).round().clamp(-1, 1).to(torch.int8)
    return codes.reshape(weight.shape[0], -1)[:, : weight.shape[1]], effective_scales


def transaction_schedule(
    step: int,
    total_steps: int,
    *,
    compress_ratio: float = 0.25,
    initial_temperature: float = 0.3508855606815209,
) -> Tuple[float, float]:
    """Pressure ramp followed by cosine temperature hardening."""
    if total_steps <= 0 or step >= total_steps:
        return 1.0, 0.0
    ratio = min(max(step / total_steps, 0.0), 1.0)
    pressure = min(ratio / max(compress_ratio, 1e-12), 1.0)
    if ratio <= compress_ratio:
        temperature = initial_temperature
    else:
        phase = (ratio - compress_ratio) / max(1.0 - compress_ratio, 1e-12)
        temperature = initial_temperature * 0.5 * (1.0 + math.cos(math.pi * phase))
    return pressure, temperature


def q2_g128_physical_bpw(group_size: int = 128, scale_bits: int = 16) -> float:
    """Physical bpw for two-bit slots plus one group scale."""
    return 2.0 + scale_bits / group_size


@torch.no_grad()
def weighted_symmetric_q4_project(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    group_size: int = 128,
    iterations: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted symmetric signed-INT4 projection per input group.

    Codes use the deployable signed range ``[-8, 7]``.  Alternating rounding
    and weighted least-squares scale updates make this a strong Q4-g128 rescue
    baseline without introducing per-value metadata.
    """
    if input_second_moment.ndim != 1 or input_second_moment.numel() != weight.shape[1]:
        raise ValueError("input_second_moment must match weight input features")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    grouped, padding, size = padded_grouped(weight.detach(), group_size)
    moment = input_second_moment.detach().float().clamp_min(0)
    if padding:
        moment = F.pad(moment, (0, padding))
    moment = moment.view(1, -1, size).expand_as(grouped)
    scale = grouped.abs().amax(-1).div(7.0).clamp_min(1e-5)
    for _ in range(iterations):
        codes = (grouped / scale.unsqueeze(-1)).round().clamp(-8, 7)
        denominator = (moment * codes.square()).sum(-1)
        fitted = (moment * codes * grouped).sum(-1).div(
            denominator.clamp_min(1e-12)
        )
        scale = torch.where(
            denominator > 0, fitted.abs().clamp_min(1e-5), scale
        )
    codes = (grouped / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    error = (
        moment * (grouped - codes.float() * scale.unsqueeze(-1)).square()
    ).sum(-1)
    return codes, scale, error


def q4_g128_physical_bpw(group_size: int = 128, scale_bits: int = 16) -> float:
    """Physical bpw for four-bit codes plus one group scale."""
    return 4.0 + scale_bits / group_size
