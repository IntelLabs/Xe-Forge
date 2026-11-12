"""Bytes per stored element by dtype name; packed types carry their packing in the shape."""

from __future__ import annotations

import math

STORAGE_BYTES = {
    "float64": 8, "int64": 8,
    "float32": 4, "int32": 4, "complex64": 8,
    "bfloat16": 2, "float16": 2, "int16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1, "float4_e2m1fn_x2": 1, "int8": 1, "uint8": 1, "bool": 1,
    # 4-bit types reported at their logical shape.
    "uint4": 0.5, "int4": 0.5, "nf4": 0.5, "float4_e2m1": 0.5,
}  # fmt: skip


ALIASES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}


def canonical(name: str) -> str:
    """A dtype by torch's name, short aliases accepted; ValueError when sizes do not know it."""
    dtype = ALIASES.get(name, name)
    if dtype not in STORAGE_BYTES:
        raise ValueError(f"{name!r} is not a dtype ({', '.join([*ALIASES, *STORAGE_BYTES])})")
    return dtype


def parse_weights(text: str | None) -> tuple[str | None, str | None]:
    """``auto`` | ``<dtype>`` | ``<scheme>`` | ``<scheme>:<dtype>`` -> (scheme, dtype).

    The dtype is what code checks; the scheme is free text for the capture agent."""
    if text in (None, "", "auto"):
        return None, None
    scheme, _, dtype = text.rpartition(":")
    if not scheme and ALIASES.get(dtype, dtype) not in STORAGE_BYTES:
        return dtype, None
    return scheme or None, canonical(dtype)


def _floats(dtypes) -> list[str]:
    return [d for d in dtypes if d and d.startswith(("bfloat", "float"))]


def compute_dtype(dtypes: list, default: str | None = None) -> str | None:
    """The model's compute dtype from dtypes ordered by share: the first float of two bytes or
    more (fp8 GEMMs of a quantized model do not set it), else the first float, else
    ``default``."""
    floats = _floats(dtypes)
    wide = [d for d in floats if (storage_bytes(d) or 0) >= 2]
    return (wide or floats or [default])[0]


def observed(ops: list) -> dict[str, float]:
    """Share of op device time by the dtype of each op's first float argument, its activation
    operand (not an fp32 scale or norm weight further on)."""
    us: dict[str, float] = {}
    for op in ops:
        d = next(iter(_floats(a.dtype for a in op.args if a is not None)), None)
        if d:
            us[d] = us.get(d, 0.0) + op.device_us
    total = sum(us.values())
    return (
        {d: round(t / total, 4) for d, t in sorted(us.items(), key=lambda kv: -kv[1])}
        if total
        else {}
    )


def storage_bytes(dtype: str | None) -> float | None:
    return STORAGE_BYTES.get(dtype or "")


def tensor_bytes(shape: list[int], dtype: str | None) -> float | None:
    """Storage bytes of a tensor; None when a dim or the dtype is unknown."""
    size = storage_bytes(dtype)
    if size is None or any(not isinstance(d, int) or d < 0 for d in shape):
        return None
    return math.prod(shape) * size
