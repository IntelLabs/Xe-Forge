"""Bytes per stored element by dtype name; packed types carry their packing in the shape."""

from __future__ import annotations

import math

STORAGE_BYTES = {
    "float64": 8, "int64": 8,
    "float32": 4, "int32": 4, "complex64": 8,
    "bfloat16": 2, "float16": 2, "int16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1, "float4_e2m1fn_x2": 1, "int8": 1, "uint8": 1, "bool": 1,
    # OpenVINO reports 4-bit types at their logical shape.
    "uint4": 0.5, "int4": 0.5, "nf4": 0.5, "float4_e2m1": 0.5,
}  # fmt: skip


def storage_bytes(dtype: str | None) -> float | None:
    return STORAGE_BYTES.get(dtype or "")


def tensor_bytes(shape: list[int], dtype: str | None) -> float | None:
    """Storage bytes of a tensor; None when a dim or the dtype is unknown."""
    size = storage_bytes(dtype)
    if size is None or any(not isinstance(d, int) or d < 0 for d in shape):
        return None
    return math.prod(shape) * size
