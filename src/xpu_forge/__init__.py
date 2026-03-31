"""XPU Forge - Multi-stage optimization pipeline for Intel XPU.
Stages: Analysis -> Algorithmic -> DType -> Fusion -> Memory -> BlockPtrs -> Persistent -> XPU.
Uses LLM knowledge instead of local YAML knowledge base.
"""

from xpu_forge.config import Config, get_config, override_config
from xpu_forge.models import OptimizationResult, OptimizationStage
from xpu_forge.pipeline import XPUForgePipeline

__version__ = "0.2.0"
__all__ = [
    "Config",
    "OptimizationResult",
    "OptimizationStage",
    "XPUForgePipeline",
    "get_config",
    "override_config",
]
