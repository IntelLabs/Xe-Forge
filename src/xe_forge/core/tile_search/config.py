"""
YAML configuration parser for tile tuning workloads.

Example (FA mode):

    mode: fa
    dtype: bf16
    max_rounds: 5
    output: fa_tile_results.json

    tune-xpu:
      - name: "Gemma 3 27B"
        dims:
          head_dim: 256
          batch: 1
          num_heads_q: 16
          num_heads_kv: 16
          seq_qo: 4096
          seq_kv: 4096

Example (GEMM mode):

    mode: gemm
    dtype: bf16
    max_rounds: 5

    tune-xpu:
      - name: "Large square"
        dims:
          M: 5120
          N: 4096
          K: 4096
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TuneConfig:
    mode: str  # "fa" or "gemm"
    dtype: str = "bf16"
    max_rounds: int = 5
    output: str = "tile_tuning_results.json"
    causal: bool = False
    fa_mode: str = "prefill"  # "prefill" or "decode"
    persistent: bool = False
    workloads: list[dict] = field(default_factory=list)


def load_tune_config(path: str | Path) -> TuneConfig:
    """Load a tile tuning config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    mode = raw.get("mode", "fa")
    cfg = TuneConfig(
        mode=mode,
        dtype=raw.get("dtype", "bf16"),
        max_rounds=raw.get("max_rounds", 5),
        output=raw.get("output", "tile_tuning_results.json"),
        causal=raw.get("causal", False),
        fa_mode=raw.get("fa_mode", "prefill"),
        persistent=raw.get("persistent", False),
    )

    variant_key = None
    for key in ("tune-xpu", "tune-gpu", "tune"):
        if key in raw:
            variant_key = key
            break

    if variant_key is None:
        return cfg

    for entry in raw[variant_key]:
        dims = entry.get("dims", {})
        name = entry.get("name", "")
        workload = {**dims, "name": name}
        cfg.workloads.append(workload)

    return cfg
