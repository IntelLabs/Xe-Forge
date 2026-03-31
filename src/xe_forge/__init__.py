"""Xe Forge - Multi-stage optimization pipeline for Intel XPU.
Stages: Analysis -> Algorithmic -> DType -> Fusion -> Memory -> BlockPtrs -> Persistent -> XPU.
Uses LLM knowledge instead of local YAML knowledge base.
"""

from xe_forge.config import Config, get_config, override_config
from xe_forge.models import OptimizationResult, OptimizationStage
from xe_forge.pipeline import XeForgePipeline

__version__ = "0.2.0"
__all__ = [
    "Config",
    "OptimizationResult",
    "OptimizationStage",
    "XeForgePipeline",
    "get_config",
    "override_config",
]
