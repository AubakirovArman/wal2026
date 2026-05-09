"""Route encoding → unique-ID codebook.

A route = (digits[..,L_max], stop_depth[..]). We pack it into a single int64 key
so torch.unique can deduplicate quickly on GPU. Digits are {-1,0,+1}; we shift
them to {0,1,2} and pack base-3. stop_depth occupies the top 4 bits of the key.

Layout of the int64 key (L_max up to 12 → 3^12 = 531_441 fits in 20 bits):
  bits  0..19 : base-3 packed digits (digit 0 is least significant)
  bits 20..23 : stop_depth (0..12)
  bits 24..63 : reserved / zero
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class Codebook:
    keys: Tensor            # int32[M] — unique packed keys (sorted ascending)
    digits: Tensor          # int8[M, L_max] — decoded ternary digits for each key
    stop_depth: Tensor      # int8[M] — stop depth per key
    l_max: int

    @property
    def size(self) -> int:
        return int(self.keys.numel())


def _pack_keys(digits: Tensor, stop_depth: Tensor, l_max: int) -> Tensor:
    """Pack (digits, stop_depth) -> int64 keys with shape digits.shape[:-1]."""
    assert digits.shape[-1] == l_max
    key = torch.zeros(digits.shape[:-1], dtype=torch.int32, device=digits.device)
    for i in range(l_max):
        key += (digits[..., i].to(torch.int32) + 1) * (3 ** i)
    key = key | (stop_depth.to(torch.int32) << 20)
    return key


def _unpack_keys(keys: Tensor, l_max: int) -> Tuple[Tensor, Tensor]:
    digit_part = keys & ((1 << 20) - 1)
    stop_depth = ((keys >> 20) & 0xF).to(torch.int8)
    digits = torch.empty(
        (keys.numel(), l_max), dtype=torch.int8, device=keys.device
    )
    tmp = digit_part.clone()
    for i in range(l_max):
        digits[:, i] = (tmp % 3).to(torch.int8) - 1  # back to {-1,0,+1}
        tmp //= 3
    return digits, stop_depth


def build_codebook(digits: Tensor, stop_depth: Tensor, l_max: int) -> Tuple[Codebook, Tensor]:
    """Return (codebook, ids) where ids is int32 with shape digits.shape[:-1].

    Each element of ids indexes into codebook.keys / codebook.digits.
    """
    flat_shape = digits.shape[:-1]
    digits_flat = digits.reshape(-1, l_max)
    stop_flat = stop_depth.reshape(-1)
    keys = _pack_keys(digits_flat, stop_flat, l_max)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    unique_digits, unique_stop = _unpack_keys(unique_keys, l_max)
    cb = Codebook(
        keys=unique_keys,
        digits=unique_digits,
        stop_depth=unique_stop,
        l_max=l_max,
    )
    ids = inverse.to(torch.int32).reshape(flat_shape)
    return cb, ids


def count_code_frequencies(digits: Tensor, stop_depth: Tensor, l_max: int) -> Tuple[Tensor, Tensor]:
    keys = _pack_keys(digits.reshape(-1, l_max), stop_depth.reshape(-1), l_max)
    return torch.unique(keys, sorted=True, return_counts=True)
