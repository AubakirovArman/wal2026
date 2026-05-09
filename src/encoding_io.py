from __future__ import annotations

from pathlib import Path

import torch

from .block_vq import BlockRVQEncoding, GroupedBlockRVQEncoding


def _serialize_block_encoding(enc: BlockRVQEncoding) -> dict[str, object]:
    return {
        "stage_ids": [tensor.detach().cpu() for tensor in enc.stage_ids],
        "codebooks": [tensor.detach().cpu() for tensor in enc.codebooks],
        "stage_value_dims": list(enc.stage_value_dims),
        "stages_per_split": list(enc.stages_per_split),
        "stage_scales": None if enc.stage_scales is None else enc.stage_scales.detach().cpu(),
        "residual_correction": enc.residual_correction,
        "residual_signs": None if enc.residual_signs is None else enc.residual_signs.detach().cpu(),
        "residual_scale": None if enc.residual_scale is None else enc.residual_scale.detach().cpu(),
        "row_scale": enc.row_scale.detach().cpu(),
        "block_scale": None if enc.block_scale is None else enc.block_scale.detach().cpu(),
        "transform_kind": enc.transform_kind,
        "transform_matrix": None if enc.transform_matrix is None else enc.transform_matrix.detach().cpu(),
        "transform_bias": None if enc.transform_bias is None else enc.transform_bias.detach().cpu(),
        "product_splits": int(enc.product_splits),
        "original_shape": list(enc.original_shape),
        "padded_cols": int(enc.padded_cols),
        "block_size": int(enc.block_size),
        "sample_rel_mse": float(enc.sample_rel_mse),
    }


def _deserialize_block_encoding(payload: dict[str, object], device: torch.device | None = None) -> BlockRVQEncoding:
    def _move(tensor):
        if tensor is None or device is None:
            return tensor
        return tensor.to(device)

    return BlockRVQEncoding(
        stage_ids=tuple(_move(tensor) for tensor in payload["stage_ids"]),
        codebooks=tuple(_move(tensor) for tensor in payload["codebooks"]),
        stage_value_dims=tuple(int(x) for x in payload["stage_value_dims"]),
        stages_per_split=tuple(int(x) for x in payload["stages_per_split"]),
        stage_scales=_move(payload["stage_scales"]),
        residual_correction=str(payload["residual_correction"]),
        residual_signs=_move(payload["residual_signs"]),
        residual_scale=_move(payload["residual_scale"]),
        row_scale=_move(payload["row_scale"]),
        block_scale=_move(payload["block_scale"]),
        transform_kind=str(payload["transform_kind"]),
        transform_matrix=_move(payload["transform_matrix"]),
        transform_bias=_move(payload["transform_bias"]),
        product_splits=int(payload["product_splits"]),
        original_shape=tuple(int(x) for x in payload["original_shape"]),
        padded_cols=int(payload["padded_cols"]),
        block_size=int(payload["block_size"]),
        sample_rel_mse=float(payload["sample_rel_mse"]),
    )


def serialize_grouped_encoding(enc: GroupedBlockRVQEncoding) -> dict[str, object]:
    return {
        "groups": [_serialize_block_encoding(group) for group in enc.groups],
        "row_slices": [list(item) for item in enc.row_slices],
        "original_shape": list(enc.original_shape),
    }


def deserialize_grouped_encoding(payload: dict[str, object], device: torch.device | None = None) -> GroupedBlockRVQEncoding:
    return GroupedBlockRVQEncoding(
        groups=tuple(_deserialize_block_encoding(group, device=device) for group in payload["groups"]),
        row_slices=tuple(tuple(int(x) for x in item) for item in payload["row_slices"]),
        original_shape=tuple(int(x) for x in payload["original_shape"]),
    )


def save_grouped_encoding_map(path: str | Path, encodings: dict[str, GroupedBlockRVQEncoding]) -> Path:
    target = Path(path)
    payload = {
        "version": 1,
        "encodings": {name: serialize_grouped_encoding(enc) for name, enc in encodings.items()},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_grouped_encoding_map(path: str | Path, device: torch.device | None = None) -> dict[str, GroupedBlockRVQEncoding]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"unsupported encoding map version: {payload.get('version')}")
    return {
        name: deserialize_grouped_encoding(enc_payload, device=device)
        for name, enc_payload in payload["encodings"].items()
    }