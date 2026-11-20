"""Xe Forge - Multi-stage optimization pipeline for Intel XPU.
Stages: Analysis -> Algorithmic -> DType -> Fusion -> Memory -> BlockPtrs -> Persistent -> XPU.
Uses LLM knowledge instead of local YAML knowledge base.
"""

__version__ = "0.3.0"
__all__ = [
    "Config",
    "OptimizationResult",
    "OptimizationStage",
    "XeForgePipeline",
    "get_config",
    "override_config",
]


def load_env() -> None:
    """Load ``.env`` before anything reads the environment: the working directory's, then
    the checkout's; a variable already set wins. A framework's venv may lack python-dotenv."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv()


# Lazy: the capture tools run inside a framework's venv and must not pull in the config stack.
_LAZY = {
    "Config": "xe_forge.config",
    "get_config": "xe_forge.config",
    "override_config": "xe_forge.config",
    "OptimizationResult": "xe_forge.models",
    "OptimizationStage": "xe_forge.models",
    "XeForgePipeline": "xe_forge.pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
