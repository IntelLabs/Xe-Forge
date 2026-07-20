"""Xe Forge - Multi-stage optimization pipeline for Intel XPU.
Stages: Analysis -> Algorithmic -> DType -> Fusion -> Memory -> BlockPtrs -> Persistent -> XPU.
Uses LLM knowledge instead of local YAML knowledge base.
"""

from xe_forge.config import Config, get_config, override_config
from xe_forge.models import OptimizationResult, OptimizationStage

__version__ = "0.2.0"
__all__ = [
    "Config",
    "OptimizationResult",
    "OptimizationStage",
    "XeForgePipeline",
    "get_config",
    "override_config",
]


def __getattr__(name: str):
    if name == "XeForgePipeline":
        from xe_forge.pipeline import XeForgePipeline

        return XeForgePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
