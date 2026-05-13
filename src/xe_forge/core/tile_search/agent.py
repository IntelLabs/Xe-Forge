"""
LLM-driven tile configuration tuning agent for Intel Xe kernels.

Uses a strategy pattern to support different kernel types (GEMM, FA, etc.)
through a single propose-validate-benchmark loop. The LLM proposes tile
configurations as structured data, which are validated against hardware
constraints, compiled into a C++ template, benchmarked, and fed back
for the next round.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import dspy

from xe_forge.core.tile_search.templates import (
    generate_fa_v2_source,
    generate_gemm_source,
    generate_grouped_gemm_source,
    generate_moe_gemm_source,
)
from xe_forge.core.tile_search.validators import (
    KNOWN_FA_CONFIGS,
    FATileConfig,
    validate_and_derive,
    validate_fa_tile,
)
from xe_forge.core.tile_search.validators.gemm import DTYPE_BITS
from xe_forge.models import TileBenchResult, TileConfig, TileTuningResult

if TYPE_CHECKING:
    from xe_forge.core.sycl_executor import SyclExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class GEMMTileTuningSignature(dspy.Signature):
    """Propose CUTLASS GEMM tile configurations for Intel Xe GPU.

    You are an expert in Intel Xe GPU architecture and GEMM tiling optimization.
    Given a problem shape (M, N, K), hardware constraints, and the history of
    previously tested configurations with their measured performance, propose
    new tile configurations that are likely to maximize TFLOPS.

    HARDWARE CONSTRAINTS (Intel Xe / BMG):
    - Workgroup tile shape: [wg_M, wg_N, wg_K]
    - DPAS atom shape: M=8, N=16, K depends on dtype (16 for bf16/f16)
    - wg_M must be divisible by 8
    - wg_N must be divisible by 16
    - wg_K must be divisible by atom_K (16 for bf16)
    - Subgroup layout is auto-derived: total subgroups <= 32
    - SLM (shared local memory): 128 KB on BMG
    - Typical good wg_K values: 16, 32, 64
    - Typical good wg_M/wg_N values: 16, 32, 64, 128, 256, 512

    STRATEGY:
    - For large square problems (M,N >> 1024): try large tiles (256x256, 512x256)
    - For skinny problems (small M or N): try asymmetric tiles matching the shape
    - Higher K trades register pressure for better arithmetic intensity
    - Look at which configs performed well in history and try nearby variations
    - Avoid repeating configs already tested

    OUTPUT FORMAT:
    Return a JSON list of tile configs. Each config is a dict with keys
    "wg_M", "wg_N", "wg_K" (all integers). Propose 3-5 configs per round.
    """

    problem_shape: str = dspy.InputField(desc="GEMM dimensions as 'M=<val>, N=<val>, K=<val>'")
    hardware_info: str = dspy.InputField(desc="GPU architecture, DPAS atom shape, and constraints")
    history: str = dspy.InputField(
        desc="Previously tested configs with TFLOPS results (JSON). Empty list if first round."
    )
    default_configs: str = dspy.InputField(
        desc="Known-good default tile configs for reference (JSON list of [M,N,K])"
    )
    proposed_configs: str = dspy.OutputField(
        desc='JSON list of {"wg_M": int, "wg_N": int, "wg_K": int} dicts. Propose 3-5 new configs.'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for why these configs should work well."
    )


class GroupedGEMMTileTuningSignature(dspy.Signature):
    """Propose CUTLASS Grouped GEMM tile configurations for Intel Xe GPU.

    Same tiling constraints as standard GEMM, but the kernel fuses multiple
    GEMMs into one launch using a persistent group tile scheduler.
    Tile shapes that work well for standard GEMM are a good starting point,
    but grouped GEMM may prefer different shapes due to the scheduler overhead
    and varying group sizes.

    OUTPUT FORMAT:
    Return a JSON list of tile configs. Each config is a dict with keys
    "wg_M", "wg_N", "wg_K" (all integers). Propose 3-5 configs per round.
    """

    problem_shape: str = dspy.InputField(
        desc="Grouped GEMM dims: M=<val>, N=<val>, K=<val>, groups=<val>"
    )
    hardware_info: str = dspy.InputField(desc="GPU architecture and constraints")
    history: str = dspy.InputField(
        desc="Previously tested configs with TFLOPS results (JSON). Empty list if first round."
    )
    default_configs: str = dspy.InputField(
        desc="Known-good default tile configs for reference (JSON list of [M,N,K])"
    )
    proposed_configs: str = dspy.OutputField(
        desc='JSON list of {"wg_M": int, "wg_N": int, "wg_K": int} dicts. Propose 3-5 new configs.'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for why these configs should work well."
    )


class MoEGEMMTileTuningSignature(dspy.Signature):
    """Propose CUTLASS MoE GEMM tile configurations for Intel Xe GPU.

    MoE (Mixture of Experts) GEMM fuses multiple expert GEMMs with varying M
    dimensions into one persistent kernel. Each expert processes a different
    number of tokens (M), while N and K are shared across all experts.

    The tile scheduler distributes work across experts dynamically.
    Tile shapes with smaller wg_M may improve M-occupancy when some experts
    have few tokens. Larger tiles improve throughput for experts with many tokens.

    HARDWARE CONSTRAINTS:
    Same as standard GEMM — DPAS atom M=8, N=16, K depends on dtype.
    wg_M divisible by 8, wg_N by 16, wg_K by atom_K.

    OUTPUT FORMAT:
    Return a JSON list of tile configs. Each config is a dict with keys
    "wg_M", "wg_N", "wg_K" (all integers). Propose 3-5 configs per round.
    """

    problem_shape: str = dspy.InputField(
        desc="MoE GEMM dims: N=<val>, K=<val>, num_experts=<val>, total_tokens=<val>"
    )
    hardware_info: str = dspy.InputField(desc="GPU architecture and constraints")
    history: str = dspy.InputField(
        desc="Previously tested configs with TFLOPS results (JSON). Empty list if first round."
    )
    default_configs: str = dspy.InputField(
        desc="Known-good default tile configs for reference (JSON list of [M,N,K])"
    )
    proposed_configs: str = dspy.OutputField(
        desc='JSON list of {"wg_M": int, "wg_N": int, "wg_K": int} dicts. Propose 3-5 new configs.'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for why these configs should work well."
    )


class FATileTuningSignature(dspy.Signature):
    """Propose Flash Attention V2 tile configurations for Intel Xe GPU.

    You are an expert in Intel Xe GPU architecture and Flash Attention tiling.
    The FA kernel performs two GEMMs per attention head:
      1. S = Q * K^T  -- score computation (ShapeQK tile)
      2. O = softmax(S) * V  -- value aggregation (ShapePV tile)

    TILE PARAMETERS:
    - ShapeQK = (qk_m, qk_n, qk_k): Q-rows, K-cols, inner-K per step
    - ShapePV = (pv_m, pv_n, pv_k): must have pv_m == qk_m (shared Q dim)
    - sg_q: number of subgroups partitioning the Q dimension
    - VTiles = head_dim / pv_n must be integer

    HARDWARE CONSTRAINTS (Intel Xe / BMG):
    - DPAS atom: M=8, N=16
    - qk_m must be divisible by sg_q, and sg_tile_q = qk_m/sg_q must work with DPAS
    - qk_n must be divisible by 16
    - sg_q <= 32 (max subgroups per workgroup)
    - SLM = 128 KB
    - Typical qk_k values: 16, 32, 64
    - pipeline_stages = 2 for prefill

    KNOWN-GOOD CONFIGS (from sycl-tla):
    - HD=64:  qk=(128,64,32), pv=(128,32,64), sg=8, stages=2
    - HD=96:  qk=(128,64,32), pv=(128,32,64), sg=8, stages=2
    - HD=128: qk=(256,32,32), pv=(256,32,32), sg=16, stages=2
    - HD=192: qk=(256,64,32), pv=(256,32,64), sg=32, stages=2

    OUTPUT: JSON list of FA tile configs with keys:
    "qk_m", "qk_n", "qk_k", "pv_n", "pv_k", "sg_q", "pipeline_stages"
    """

    problem_shape: str = dspy.InputField(
        desc="FA dimensions: head_dim, batch, num_heads_q, num_heads_kv, seq_qo, seq_kv"
    )
    hardware_info: str = dspy.InputField(desc="GPU architecture and constraints")
    history: str = dspy.InputField(
        desc="Previously tested configs with TFLOPS results (JSON). Empty list if first round."
    )
    known_configs: str = dspy.InputField(
        desc="Known-good tile configs for reference head_dims (JSON)"
    )
    proposed_configs: str = dspy.OutputField(
        desc="JSON list of FA tile config dicts. Propose 3-5 new configs."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for why these configs should work well."
    )


# ---------------------------------------------------------------------------
# KernelStrategy protocol
# ---------------------------------------------------------------------------


class KernelStrategy(Protocol):
    """Pluggable strategy for kernel-specific tile tuning behavior."""

    def get_signature(self) -> type: ...
    def build_problem_str(self, workload: dict) -> str: ...
    def build_hardware_info(self, dtype: str) -> str: ...
    def get_reference_configs(self) -> str: ...
    def build_propose_kwargs(
        self, problem_str: str, hw_info: str, history_json: str
    ) -> dict[str, str]: ...
    def enrich_config(self, cfg: dict, workload: dict) -> dict: ...
    def parse_proposed(self, raw_configs: list[dict]) -> list[dict]: ...
    def validate(self, cfg: dict, dtype: str) -> tuple[bool, list[str], dict]: ...
    def generate_source(self, cfg: dict, dtype: str) -> str: ...
    def build_run_args(self, cfg: dict, workload: dict) -> dict[str, Any]: ...
    def config_key(self, cfg: dict) -> tuple: ...
    def to_tile_config(self, cfg: dict, derived: dict | None = None) -> TileConfig: ...
    def output_name(self, cfg: dict) -> str: ...
    def make_history_entry(
        self, cfg: dict, bench: TileBenchResult, derived: dict | None = None
    ) -> dict: ...
    def get_seed_config(self, workload: dict) -> dict | None: ...


# ---------------------------------------------------------------------------
# GEMMStrategy
# ---------------------------------------------------------------------------


class GEMMStrategy:
    """Tile tuning strategy for CUTLASS GEMM kernels."""

    DEFAULT_TILE_CONFIGS: ClassVar[list[list[int]]] = [
        [256, 128, 64],
        [128, 256, 64],
        [128, 128, 64],
        [256, 64, 64],
        [64, 256, 64],
        [32, 32, 64],
        [16, 64, 64],
        [64, 16, 64],
        [512, 128, 32],
        [256, 256, 32],
        [256, 128, 32],
        [256, 64, 32],
        [128, 256, 32],
        [128, 64, 32],
        [32, 32, 32],
        [16, 64, 32],
        [64, 128, 32],
        [128, 128, 32],
        [256, 256, 16],
    ]

    def __init__(self, layout_a: str = "RowMajor", layout_b: str = "RowMajor"):
        self.layout_a = layout_a
        self.layout_b = layout_b

    def get_signature(self) -> type:
        return GEMMTileTuningSignature

    def build_problem_str(self, workload: dict) -> str:
        return f"M={workload['M']}, N={workload['N']}, K={workload['K']}"

    def build_hardware_info(self, dtype: str) -> str:
        atom_k = 256 // DTYPE_BITS.get(dtype, 16)
        return (
            "Intel Xe (BMG/Battlemage). "
            f"DPAS atom shape: M=8, N=16, K={atom_k} (for {dtype}). "
            f"MaxSubgroups=32. SLM=128KB. "
            f"wg_M must be divisible by 8, wg_N by 16, wg_K by {atom_k}. "
            f"dtype={dtype}, layout_a={self.layout_a}, layout_b={self.layout_b}."
        )

    def get_reference_configs(self) -> str:
        return json.dumps(self.DEFAULT_TILE_CONFIGS)

    def build_propose_kwargs(
        self, problem_str: str, hw_info: str, history_json: str
    ) -> dict[str, str]:
        return {
            "problem_shape": problem_str,
            "hardware_info": hw_info,
            "history": history_json,
            "default_configs": self.get_reference_configs(),
        }

    def enrich_config(self, cfg: dict, workload: dict) -> dict:
        return cfg

    def parse_proposed(self, raw_configs: list[dict]) -> list[dict]:
        valid = []
        for cfg in raw_configs:
            if isinstance(cfg, dict) and "wg_M" in cfg and "wg_N" in cfg and "wg_K" in cfg:
                valid.append(
                    {"wg_M": int(cfg["wg_M"]), "wg_N": int(cfg["wg_N"]), "wg_K": int(cfg["wg_K"])}
                )
            else:
                logger.warning("Skipping malformed GEMM config: %s", cfg)
        return valid

    def validate(self, cfg: dict, dtype: str) -> tuple[bool, list[str], dict]:
        wg = [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]]
        result = validate_and_derive(wg, dtype=dtype)
        derived = {"sg": result.sg_layout} if result.valid else {}
        return result.valid, result.errors, derived

    def generate_source(self, cfg: dict, dtype: str) -> str:
        return generate_gemm_source(
            cfg["wg_M"],
            cfg["wg_N"],
            cfg["wg_K"],
            dtype=dtype,
            layout_a=self.layout_a,
            layout_b=self.layout_b,
        )

    def build_run_args(self, cfg: dict, workload: dict) -> dict[str, Any]:
        return {
            "m": workload["M"],
            "n": workload["N"],
            "k": workload["K"],
            "iterations": 20,
            "verify": 1,
        }

    def config_key(self, cfg: dict) -> tuple:
        return (cfg["wg_M"], cfg["wg_N"], cfg["wg_K"])

    def to_tile_config(self, cfg: dict, derived: dict | None = None) -> TileConfig:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return TileConfig(wg=[cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]], sg=sg)

    def output_name(self, cfg: dict) -> str:
        return f"tile_{cfg['wg_M']}x{cfg['wg_N']}x{cfg['wg_K']}"

    def make_history_entry(
        self, cfg: dict, bench: TileBenchResult, derived: dict | None = None
    ) -> dict:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return {
            "wg": [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]],
            "sg": sg,
            "status": "ok" if bench.passed else "failed",
            "tflops": bench.tflops,
            "time_ms": bench.time_ms,
            "error": bench.error,
        }

    def get_seed_config(self, workload: dict) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# FAStrategy
# ---------------------------------------------------------------------------


class FAStrategy:
    """Tile tuning strategy for Flash Attention V2 kernels."""

    def get_signature(self) -> type:
        return FATileTuningSignature

    def build_problem_str(self, workload: dict) -> str:
        return (
            f"head_dim={workload.get('head_dim', 128)}, "
            f"batch={workload.get('batch', 1)}, "
            f"num_heads_q={workload.get('num_heads_q', 32)}, "
            f"num_heads_kv={workload.get('num_heads_kv', 8)}, "
            f"seq_qo={workload.get('seq_qo', 4096)}, "
            f"seq_kv={workload.get('seq_kv', 4096)}"
        )

    def build_hardware_info(self, dtype: str) -> str:
        return (
            "Intel Xe (BMG/Battlemage). "
            "DPAS atom: M=8, N=16. MaxSubgroups=32. SLM=128KB. "
            "qk_n must be divisible by 16. "
            "qk_k typical values: 16, 32, 64. "
            "sg_q must divide qk_m, sg_q <= 32. "
            f"dtype={dtype}."
        )

    def get_reference_configs(self) -> str:
        known = {
            str(hd): {
                "qk": [c.qk_m, c.qk_n, c.qk_k],
                "pv": [c.pv_m, c.pv_n, c.pv_k],
                "sg_q": c.sg_q,
                "pipeline_stages": c.pipeline_stages,
            }
            for hd, c in KNOWN_FA_CONFIGS.items()
        }
        return json.dumps(known)

    def build_propose_kwargs(
        self, problem_str: str, hw_info: str, history_json: str
    ) -> dict[str, str]:
        return {
            "problem_shape": problem_str,
            "hardware_info": hw_info,
            "history": history_json,
            "known_configs": self.get_reference_configs(),
        }

    def enrich_config(self, cfg: dict, workload: dict) -> dict:
        if "head_dim" not in cfg and "head_dim" in workload:
            cfg["head_dim"] = workload["head_dim"]
        return cfg

    def parse_proposed(self, raw_configs: list[dict]) -> list[dict]:
        required = {"qk_m", "qk_n", "qk_k", "pv_n", "pv_k", "sg_q"}
        optional = {"pipeline_stages"}
        valid = []
        for cfg in raw_configs:
            if isinstance(cfg, dict) and required.issubset(cfg.keys()):
                entry = {k: int(cfg[k]) for k in required}
                for k in optional:
                    if k in cfg:
                        entry[k] = int(cfg[k])
                valid.append(entry)
            else:
                logger.warning("Skipping malformed FA config: %s", cfg)
        return valid

    def validate(self, cfg: dict, dtype: str) -> tuple[bool, list[str], dict]:
        head_dim = cfg.get("head_dim", 128)
        fa_cfg = FATileConfig(
            qk_m=cfg["qk_m"],
            qk_n=cfg["qk_n"],
            qk_k=cfg["qk_k"],
            pv_m=cfg["qk_m"],
            pv_n=cfg["pv_n"],
            pv_k=cfg["pv_k"],
            head_dim=head_dim,
            sg_q=cfg["sg_q"],
            pipeline_stages=cfg.get("pipeline_stages", 2),
        )
        result = validate_fa_tile(fa_cfg)
        derived = {"fa_cfg": fa_cfg} if result.valid else {}
        return result.valid, result.errors, derived

    def generate_source(self, cfg: dict, dtype: str) -> str:
        head_dim = cfg.get("head_dim", 128)
        return generate_fa_v2_source(
            wg_tile_q=cfg["qk_m"],
            wg_tile_k=cfg["qk_n"],
            wg_tile_v=cfg["pv_n"],
            sg_tile_q=cfg["qk_m"] // cfg["sg_q"],
            sg_tile_k=cfg["pv_k"],
            head_dim_qk=cfg["qk_k"],
            head_dim_v=head_dim,
            dtype=dtype,
        )

    def build_run_args(self, cfg: dict, workload: dict) -> dict[str, Any]:
        hd_qk = workload.get("head_dim_qk") or workload.get("head_dim", 128)
        hd_vo = workload.get("head_dim_vo") or workload.get("head_dim", 128)
        return {
            "batch": workload.get("batch", 1),
            "num_heads_q": workload.get("num_heads_q", 32),
            "num_heads_kv": workload.get("num_heads_kv", 8),
            "seq_len_qo": workload.get("seq_qo", 4096),
            "seq_len_kv": workload.get("seq_kv", 4096),
            "head_size_qk": hd_qk,
            "head_size_vo": hd_vo,
            "iterations": 100,
            "verify": 0,
        }

    def config_key(self, cfg: dict) -> tuple:
        return (
            cfg["qk_m"],
            cfg["qk_n"],
            cfg["qk_k"],
            cfg["pv_n"],
            cfg["pv_k"],
            cfg["sg_q"],
            cfg.get("pipeline_stages", 2),
        )

    def to_tile_config(self, cfg: dict, derived: dict | None = None) -> TileConfig:
        return TileConfig(
            wg=[cfg["qk_m"], cfg["qk_n"], cfg["qk_k"]],
            sg=[cfg["sg_q"], 1, 1],
            extra={
                "pv_n": cfg["pv_n"],
                "pv_k": cfg["pv_k"],
                "pipeline_stages": cfg.get("pipeline_stages", 2),
            },
        )

    def output_name(self, cfg: dict) -> str:
        return (
            f"fa_qk{cfg['qk_m']}x{cfg['qk_n']}x{cfg['qk_k']}"
            f"_pv{cfg['pv_n']}x{cfg['pv_k']}_sg{cfg['sg_q']}"
        )

    def make_history_entry(
        self, cfg: dict, bench: TileBenchResult, derived: dict | None = None
    ) -> dict:
        return {
            "qk": [cfg["qk_m"], cfg["qk_n"], cfg["qk_k"]],
            "pv": [cfg["qk_m"], cfg["pv_n"], cfg["pv_k"]],
            "sg_q": cfg["sg_q"],
            "pipeline_stages": cfg.get("pipeline_stages", 2),
            "status": "ok" if bench.passed else "failed",
            "tflops": bench.tflops,
            "time_ms": bench.time_ms,
            "error": bench.error,
        }

    def get_seed_config(self, workload: dict) -> dict | None:
        head_dim = workload.get("head_dim", 128)
        seed = KNOWN_FA_CONFIGS.get(head_dim)
        if seed is None:
            return None
        return {
            "qk_m": seed.qk_m,
            "qk_n": seed.qk_n,
            "qk_k": seed.qk_k,
            "pv_n": seed.pv_n,
            "pv_k": seed.pv_k,
            "sg_q": seed.sg_q,
            "pipeline_stages": seed.pipeline_stages,
            "head_dim": head_dim,
        }


# ---------------------------------------------------------------------------
# GroupedGEMMStrategy
# ---------------------------------------------------------------------------


class GroupedGEMMStrategy:
    """Tile tuning strategy for CUTLASS Grouped GEMM kernels."""

    DEFAULT_TILE_CONFIGS: ClassVar[list[list[int]]] = [
        [256, 256, 32],
        [256, 128, 32],
        [128, 256, 32],
        [128, 128, 32],
        [256, 64, 32],
        [64, 256, 32],
        [256, 128, 64],
        [128, 256, 64],
        [128, 128, 64],
        [256, 256, 16],
    ]

    def __init__(self, layout_a: str = "RowMajor", layout_b: str = "RowMajor"):
        self.layout_a = layout_a
        self.layout_b = layout_b

    def get_signature(self) -> type:
        return GroupedGEMMTileTuningSignature

    def build_problem_str(self, workload: dict) -> str:
        return (
            f"M={workload['M']}, N={workload['N']}, K={workload['K']}, "
            f"groups={workload.get('groups', 2)}"
        )

    def build_hardware_info(self, dtype: str) -> str:
        atom_k = 256 // DTYPE_BITS.get(dtype, 16)
        return (
            "Intel Xe (BMG/Battlemage). "
            f"DPAS atom shape: M=8, N=16, K={atom_k} (for {dtype}). "
            f"MaxSubgroups=32. SLM=128KB. "
            f"wg_M must be divisible by 8, wg_N by 16, wg_K by {atom_k}. "
            f"Grouped GEMM uses persistent group tile scheduler. "
            f"dtype={dtype}, layout_a={self.layout_a}, layout_b={self.layout_b}."
        )

    def get_reference_configs(self) -> str:
        return json.dumps(self.DEFAULT_TILE_CONFIGS)

    def build_propose_kwargs(
        self, problem_str: str, hw_info: str, history_json: str
    ) -> dict[str, str]:
        return {
            "problem_shape": problem_str,
            "hardware_info": hw_info,
            "history": history_json,
            "default_configs": self.get_reference_configs(),
        }

    def enrich_config(self, cfg: dict, workload: dict) -> dict:
        return cfg

    def parse_proposed(self, raw_configs: list[dict]) -> list[dict]:
        valid = []
        for cfg in raw_configs:
            if isinstance(cfg, dict) and "wg_M" in cfg and "wg_N" in cfg and "wg_K" in cfg:
                valid.append(
                    {"wg_M": int(cfg["wg_M"]), "wg_N": int(cfg["wg_N"]), "wg_K": int(cfg["wg_K"])}
                )
            else:
                logger.warning("Skipping malformed Grouped GEMM config: %s", cfg)
        return valid

    def validate(self, cfg: dict, dtype: str) -> tuple[bool, list[str], dict]:
        wg = [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]]
        result = validate_and_derive(wg, dtype=dtype)
        derived = {"sg": result.sg_layout} if result.valid else {}
        return result.valid, result.errors, derived

    def generate_source(self, cfg: dict, dtype: str) -> str:
        wg = [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]]
        result = validate_and_derive(wg, dtype=dtype)
        return generate_grouped_gemm_source(
            cfg["wg_M"],
            cfg["wg_N"],
            cfg["wg_K"],
            sg_m=result.sg_m,
            sg_n=result.sg_n,
            dtype=dtype,
            layout_a=self.layout_a,
            layout_b=self.layout_b,
        )

    def build_run_args(self, cfg: dict, workload: dict) -> dict[str, Any]:
        return {
            "m": workload["M"],
            "n": workload["N"],
            "k": workload["K"],
            "groups": workload.get("groups", 2),
            "iterations": 20,
            "verify": 1,
        }

    def config_key(self, cfg: dict) -> tuple:
        return (cfg["wg_M"], cfg["wg_N"], cfg["wg_K"])

    def to_tile_config(self, cfg: dict, derived: dict | None = None) -> TileConfig:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return TileConfig(wg=[cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]], sg=sg)

    def output_name(self, cfg: dict) -> str:
        return f"grp_{cfg['wg_M']}x{cfg['wg_N']}x{cfg['wg_K']}"

    def make_history_entry(
        self, cfg: dict, bench: TileBenchResult, derived: dict | None = None
    ) -> dict:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return {
            "wg": [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]],
            "sg": sg,
            "status": "ok" if bench.passed else "failed",
            "tflops": bench.tflops,
            "time_ms": bench.time_ms,
            "error": bench.error,
        }

    def get_seed_config(self, workload: dict) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# MoEGEMMStrategy
# ---------------------------------------------------------------------------


class MoEGEMMStrategy:
    """Tile tuning strategy for CUTLASS MoE GEMM kernels."""

    DEFAULT_TILE_CONFIGS: ClassVar[list[list[int]]] = [
        [256, 128, 32],
        [128, 128, 32],
        [256, 256, 32],
        [128, 256, 32],
        [256, 64, 32],
        [64, 128, 32],
        [256, 128, 64],
        [128, 128, 64],
    ]

    def get_signature(self) -> type:
        return MoEGEMMTileTuningSignature

    def build_problem_str(self, workload: dict) -> str:
        return (
            f"N={workload['N']}, K={workload['K']}, "
            f"num_experts={workload.get('num_experts', 8)}, "
            f"total_tokens={workload.get('total_tokens', 4096)}"
        )

    def build_hardware_info(self, dtype: str) -> str:
        atom_k = 256 // DTYPE_BITS.get(dtype, 16)
        return (
            "Intel Xe (BMG/Battlemage). "
            f"DPAS atom shape: M=8, N=16, K={atom_k} (for {dtype}). "
            f"MaxSubgroups=32. SLM=128KB. "
            f"wg_M must be divisible by 8, wg_N by 16, wg_K by {atom_k}. "
            f"MoE GEMM: each expert has different M (token count). "
            f"Smaller wg_M improves M-occupancy for small experts. "
            f"dtype={dtype}."
        )

    def get_reference_configs(self) -> str:
        return json.dumps(self.DEFAULT_TILE_CONFIGS)

    def build_propose_kwargs(
        self, problem_str: str, hw_info: str, history_json: str
    ) -> dict[str, str]:
        return {
            "problem_shape": problem_str,
            "hardware_info": hw_info,
            "history": history_json,
            "default_configs": self.get_reference_configs(),
        }

    def enrich_config(self, cfg: dict, workload: dict) -> dict:
        return cfg

    def parse_proposed(self, raw_configs: list[dict]) -> list[dict]:
        valid = []
        for cfg in raw_configs:
            if isinstance(cfg, dict) and "wg_M" in cfg and "wg_N" in cfg and "wg_K" in cfg:
                valid.append(
                    {"wg_M": int(cfg["wg_M"]), "wg_N": int(cfg["wg_N"]), "wg_K": int(cfg["wg_K"])}
                )
            else:
                logger.warning("Skipping malformed MoE GEMM config: %s", cfg)
        return valid

    def validate(self, cfg: dict, dtype: str) -> tuple[bool, list[str], dict]:
        wg = [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]]
        result = validate_and_derive(wg, dtype=dtype)
        derived = {"sg": result.sg_layout} if result.valid else {}
        return result.valid, result.errors, derived

    def generate_source(self, cfg: dict, dtype: str) -> str:
        wg = [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]]
        result = validate_and_derive(wg, dtype=dtype)
        return generate_moe_gemm_source(
            cfg["wg_M"],
            cfg["wg_N"],
            cfg["wg_K"],
            sg_m=result.sg_m,
            sg_n=result.sg_n,
            dtype=dtype,
        )

    def build_run_args(self, cfg: dict, workload: dict) -> dict[str, Any]:
        return {
            "n": workload["N"],
            "k": workload["K"],
            "num_experts": workload.get("num_experts", 8),
            "total_tokens": workload.get("total_tokens", 4096),
            "iterations": 20,
            "verify": 0,
        }

    def config_key(self, cfg: dict) -> tuple:
        return (cfg["wg_M"], cfg["wg_N"], cfg["wg_K"])

    def to_tile_config(self, cfg: dict, derived: dict | None = None) -> TileConfig:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return TileConfig(wg=[cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]], sg=sg)

    def output_name(self, cfg: dict) -> str:
        return f"moe_{cfg['wg_M']}x{cfg['wg_N']}x{cfg['wg_K']}"

    def make_history_entry(
        self, cfg: dict, bench: TileBenchResult, derived: dict | None = None
    ) -> dict:
        sg = derived.get("sg", [0, 0, 1]) if derived else [0, 0, 1]
        return {
            "wg": [cfg["wg_M"], cfg["wg_N"], cfg["wg_K"]],
            "sg": sg,
            "status": "ok" if bench.passed else "failed",
            "tflops": bench.tflops,
            "time_ms": bench.time_ms,
            "error": bench.error,
        }

    def get_seed_config(self, workload: dict) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# TileTuningAgent
# ---------------------------------------------------------------------------


class TileTuningAgent:
    """Orchestrates the propose-validate-benchmark loop for tile tuning.

    Kernel-specific behavior is injected via a KernelStrategy.
    """

    def __init__(
        self,
        executor: SyclExecutor,
        strategy: KernelStrategy,
        dtype: str = "bf16",
    ):
        self.executor = executor
        self.strategy = strategy
        self.dtype = dtype
        self.predictor = dspy.Predict(strategy.get_signature())

    def tune(self, workload: dict, max_rounds: int = 5) -> TileTuningResult:
        """Run the tile tuning loop for a workload.

        Args:
            workload: Dimension dict (GEMM: {M,N,K}, FA: {head_dim,batch,...})
            max_rounds: Maximum LLM proposal rounds.
        """
        history: list[dict] = []
        all_results: list[TileBenchResult] = []
        best_config: TileConfig | None = None
        best_tflops: float = 0.0
        best_time_ms: float | None = None
        rounds_without_improvement = 0
        tested_keys: set[tuple] = set()

        problem_str = self.strategy.build_problem_str(workload)
        hw_info = self.strategy.build_hardware_info(self.dtype)

        logger.info("Starting tile tuning for %s", problem_str)

        # Seed with known-good config if available
        seed_cfg = self.strategy.get_seed_config(workload)
        if seed_cfg is not None:
            logger.info("Seeding with known config")
            bench = self._benchmark(seed_cfg, workload)
            all_results.append(bench)
            entry = self.strategy.make_history_entry(seed_cfg, bench)
            history.append(entry)
            tested_keys.add(self.strategy.config_key(seed_cfg))
            if bench.passed and bench.tflops and bench.tflops > best_tflops:
                best_tflops = bench.tflops
                best_time_ms = bench.time_ms
                best_config = self.strategy.to_tile_config(seed_cfg)
                logger.info("Seed baseline: %.2f TFLOPS", best_tflops)

        for round_idx in range(max_rounds):
            logger.info("Round %d/%d (best: %.2f TFLOPS)", round_idx + 1, max_rounds, best_tflops)

            proposed = self._propose(problem_str, hw_info, history)
            if not proposed:
                logger.warning("LLM returned no configs in round %d", round_idx + 1)
                rounds_without_improvement += 1
                if rounds_without_improvement >= 2:
                    break
                continue

            improved = False

            for cfg in proposed:
                cfg = self.strategy.enrich_config(cfg, workload)

                key = self.strategy.config_key(cfg)
                if key in tested_keys:
                    logger.debug("Skipping already-tested config %s", key)
                    continue
                tested_keys.add(key)

                valid, errors, derived = self.strategy.validate(cfg, self.dtype)
                if not valid:
                    history.append({"key": key, "status": "invalid", "errors": errors})
                    all_results.append(
                        TileBenchResult(
                            config=self.strategy.to_tile_config(cfg),
                            error="; ".join(errors),
                        )
                    )
                    logger.info("Invalid config %s: %s", key, errors)
                    continue

                bench = self._benchmark(cfg, workload)
                all_results.append(bench)
                entry = self.strategy.make_history_entry(cfg, bench, derived)
                entry["key"] = key
                history.append(entry)

                if bench.passed and bench.tflops and bench.tflops > best_tflops:
                    best_tflops = bench.tflops
                    best_time_ms = bench.time_ms
                    best_config = self.strategy.to_tile_config(cfg, derived)
                    improved = True
                    logger.info(
                        "New best: %s -> %.2f TFLOPS (%.4f ms)", key, best_tflops, best_time_ms
                    )

            if improved:
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= 2:
                    logger.info("No improvement for 2 rounds, stopping early")
                    break

        logger.info(
            "Tile tuning complete: tested %d configs, best=%.2f TFLOPS",
            len(all_results),
            best_tflops,
        )

        return TileTuningResult(
            problem_shape=workload,
            configs_tested=all_results,
            best_config=best_config,
            best_tflops=best_tflops if best_tflops > 0 else None,
            best_time_ms=best_time_ms,
        )

    def _propose(self, problem_str: str, hw_info: str, history: list[dict]) -> list[dict]:
        try:
            history_json = json.dumps(history, default=str)
            kwargs = self.strategy.build_propose_kwargs(problem_str, hw_info, history_json)
            result = self.predictor(**kwargs)

            raw = result.proposed_configs
            if isinstance(raw, str):
                configs = json.loads(raw)
            elif isinstance(raw, list):
                configs = raw
            else:
                logger.warning("Unexpected proposed_configs type: %s", type(raw))
                return []

            parsed = self.strategy.parse_proposed(configs)

            if result.reasoning:
                logger.info("LLM reasoning: %s", result.reasoning[:200])

            return parsed

        except Exception as e:
            logger.error("LLM proposal failed: %s", e)
            return []

    def _benchmark(self, cfg: dict, workload: dict) -> TileBenchResult:
        source = self.strategy.generate_source(cfg, self.dtype)
        name = self.strategy.output_name(cfg)
        run_args = self.strategy.build_run_args(cfg, workload)

        result = self.executor.execute_raw(
            kernel_code=source,
            output_name=name,
            args=run_args,
        )

        tile_config = self.strategy.to_tile_config(cfg)

        if not result.success:
            return TileBenchResult(
                config=tile_config, error=result.error_message or "Unknown error"
            )

        return TileBenchResult(
            config=tile_config,
            time_ms=result.execution_time_ms,
            tflops=result.tflops,
            passed=result.output_correct is not False,
        )


def export_results_json(results: list[TileTuningResult]) -> str:
    """Export tuning results as JSON compatible with SYCL_TLA_ADDITIONAL_TILE_SHAPES."""
    entries = []
    for r in results:
        if r.best_config is None:
            continue
        entry: dict[str, Any] = {"wg": r.best_config.wg, "sg": r.best_config.sg}
        if r.best_config.extra:
            entry["extra"] = r.best_config.extra
        entries.append(entry)
    return json.dumps(entries, indent=2)
