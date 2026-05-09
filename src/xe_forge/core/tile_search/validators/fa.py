"""
Flash Attention tile configuration validator for Intel Xe.

Validates FA tile shapes (ShapeQK, ShapePV, SubgroupLayout) against
hardware constraints. The FA kernel has different constraints than GEMM:
- Two matmuls: Q*K^T (ShapeQK) and P*V (ShapePV)
- Shared Q-tile dimension across both
- VTiles = head_dim / pv_n must be integer
- Subgroup count from QK layout, PV layout auto-derived
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DPAS_ATOM_M = 8
DPAS_ATOM_N = 16
MAX_SG = 32
SLM_BYTES = 128 * 1024


@dataclass
class FATileConfig:
    qk_m: int
    qk_n: int
    qk_k: int
    pv_m: int
    pv_n: int
    pv_k: int
    head_dim: int
    sg_q: int
    pipeline_stages: int = 2


@dataclass
class FATileValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    config: FATileConfig | None = None
    vtiles: int = 0


def validate_fa_tile(cfg: FATileConfig) -> FATileValidation:
    """Validate an FA tile configuration against Intel Xe constraints."""
    errors: list[str] = []

    if any(
        v <= 0
        for v in [
            cfg.qk_m,
            cfg.qk_n,
            cfg.qk_k,
            cfg.pv_m,
            cfg.pv_n,
            cfg.pv_k,
            cfg.head_dim,
            cfg.sg_q,
        ]
    ):
        errors.append("All dimensions must be positive")
        return FATileValidation(valid=False, errors=errors)

    if cfg.qk_m != cfg.pv_m:
        errors.append(f"qk_m ({cfg.qk_m}) must equal pv_m ({cfg.pv_m})")

    if cfg.head_dim % cfg.pv_n != 0:
        errors.append(f"head_dim ({cfg.head_dim}) must be divisible by pv_n ({cfg.pv_n})")

    vtiles = cfg.head_dim // cfg.pv_n if cfg.pv_n > 0 else 0

    sg_tile_q = cfg.qk_m // cfg.sg_q if cfg.sg_q > 0 else 0
    if cfg.qk_m % cfg.sg_q != 0:
        errors.append(f"qk_m ({cfg.qk_m}) must be divisible by sg_q ({cfg.sg_q})")

    if cfg.sg_q > MAX_SG:
        errors.append(f"sg_q ({cfg.sg_q}) exceeds MaxSG ({MAX_SG})")

    atom_m = math.gcd(DPAS_ATOM_M, sg_tile_q) if sg_tile_q > 0 else 0
    if atom_m > 0:
        if sg_tile_q % atom_m != 0:
            errors.append(f"sg_tile_q ({sg_tile_q}) not divisible by DPAS atom M ({atom_m})")

    if cfg.qk_n % DPAS_ATOM_N != 0:
        errors.append(f"qk_n ({cfg.qk_n}) must be divisible by DPAS atom N ({DPAS_ATOM_N})")

    if cfg.pv_n % DPAS_ATOM_N != 0 and cfg.pv_n % DPAS_ATOM_M != 0:
        errors.append(f"pv_n ({cfg.pv_n}) must be divisible by DPAS atom dim")

    if cfg.qk_k not in (16, 32, 64):
        errors.append(f"qk_k ({cfg.qk_k}) should be 16, 32, or 64")

    if cfg.pv_k not in (16, 32, 64, 128, 256, 512):
        errors.append(f"pv_k ({cfg.pv_k}) should be a power of 2 (16-512)")

    if cfg.pipeline_stages not in (1, 2, 3):
        errors.append(f"pipeline_stages must be 1, 2, or 3, got {cfg.pipeline_stages}")

    if errors:
        return FATileValidation(valid=False, errors=errors, config=cfg)

    return FATileValidation(valid=True, config=cfg, vtiles=vtiles)


KNOWN_FA_CONFIGS: dict[int, FATileConfig] = {
    64: FATileConfig(
        qk_m=128,
        qk_n=64,
        qk_k=32,
        pv_m=128,
        pv_n=32,
        pv_k=64,
        head_dim=64,
        sg_q=8,
    ),
    96: FATileConfig(
        qk_m=128,
        qk_n=64,
        qk_k=32,
        pv_m=128,
        pv_n=32,
        pv_k=64,
        head_dim=96,
        sg_q=8,
    ),
    128: FATileConfig(
        qk_m=256,
        qk_n=32,
        qk_k=32,
        pv_m=256,
        pv_n=32,
        pv_k=32,
        head_dim=128,
        sg_q=16,
    ),
    192: FATileConfig(
        qk_m=256,
        qk_n=64,
        qk_k=32,
        pv_m=256,
        pv_n=32,
        pv_k=64,
        head_dim=192,
        sg_q=32,
    ),
    256: FATileConfig(
        qk_m=256,
        qk_n=64,
        qk_k=32,
        pv_m=256,
        pv_n=32,
        pv_k=64,
        head_dim=256,
        sg_q=32,
        pipeline_stages=1,
    ),
    512: FATileConfig(
        qk_m=128,
        qk_n=64,
        qk_k=32,
        pv_m=128,
        pv_n=32,
        pv_k=64,
        head_dim=512,
        sg_q=32,
        pipeline_stages=1,
    ),
}
