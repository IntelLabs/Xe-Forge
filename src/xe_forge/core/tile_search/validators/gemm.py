"""
Tile configuration validator for Intel Xe GEMM kernels.

Mirrors the auto-derivation logic in sycl-tla's xe_mma_builder.inl
(CollectiveBuilder) to validate tile shapes and compute subgroup layouts
without requiring C++ compilation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DPAS_ATOM_N = 16

DTYPE_BITS: dict[str, int] = {
    "bf16": 16,
    "f16": 16,
    "tf32": 32,
    "f32": 32,
    "int8": 8,
    "f8": 8,
}

MAX_SG = 32


@dataclass
class TileValidation:
    valid: bool
    errors: list[str]
    sg_m: int = 0
    sg_n: int = 0
    sg_k: int = 1

    @property
    def sg_layout(self) -> list[int]:
        return [self.sg_m, self.sg_n, self.sg_k]


def _dpas_atom_k(dtype: str) -> int:
    bits = DTYPE_BITS.get(dtype, 16)
    return 256 // bits


def _dpas_atom_m(wg_m: int) -> int:
    return math.gcd(8, wg_m)


def validate_tile_config(
    wg_m: int,
    wg_n: int,
    wg_k: int,
    dtype: str = "bf16",
) -> TileValidation:
    """Validate a tile configuration against Intel Xe hardware constraints.

    Mirrors xe_mma_builder.inl lines 89-98:
      DPAS_M  = gcd(8, wg_M)
      AtomGrid = TileShape / AtomShape
      SG_M0   = min(AtomGrid_M, 8)
      SG_N    = min(AtomGrid_N, MaxSG / SG_M0)
      SG_M    = min(AtomGrid_M, MaxSG / SG_N)

    Returns a TileValidation with errors (if any) and the derived sg layout.
    """
    errors: list[str] = []

    if wg_m <= 0 or wg_n <= 0 or wg_k <= 0:
        errors.append(f"Tile dimensions must be positive: got ({wg_m}, {wg_n}, {wg_k})")
        return TileValidation(valid=False, errors=errors)

    atom_m = _dpas_atom_m(wg_m)
    atom_n = DPAS_ATOM_N
    atom_k = _dpas_atom_k(dtype)

    if wg_m % atom_m != 0:
        errors.append(f"wg_M={wg_m} not divisible by DPAS atom M={atom_m}")
    if wg_n % atom_n != 0:
        errors.append(f"wg_N={wg_n} not divisible by DPAS atom N={atom_n}")
    if wg_k % atom_k != 0:
        errors.append(f"wg_K={wg_k} not divisible by DPAS atom K={atom_k}")

    if errors:
        return TileValidation(valid=False, errors=errors)

    atom_grid_m = wg_m // atom_m
    atom_grid_n = wg_n // atom_n

    sg_m0 = min(atom_grid_m, 8)
    sg_n = min(atom_grid_n, MAX_SG // sg_m0)
    sg_m = min(atom_grid_m, MAX_SG // sg_n)

    if sg_m * sg_n > MAX_SG:
        errors.append(f"sg_m*sg_n={sg_m}*{sg_n}={sg_m * sg_n} exceeds MaxSG={MAX_SG}")
        return TileValidation(valid=False, errors=errors)

    return TileValidation(valid=True, errors=[], sg_m=sg_m, sg_n=sg_n, sg_k=1)


def validate_and_derive(
    wg: list[int],
    dtype: str = "bf16",
) -> TileValidation:
    """Convenience wrapper accepting a [wg_M, wg_N, wg_K] list."""
    if len(wg) != 3:
        return TileValidation(valid=False, errors=[f"Expected 3 elements, got {len(wg)}"])
    return validate_tile_config(wg[0], wg[1], wg[2], dtype=dtype)
