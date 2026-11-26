"""X-Forge - Multi-stage optimization pipeline for Intel XPU.
Stages: Analysis -> Algorithmic -> DType -> Fusion -> Memory -> BlockPtrs -> Persistent -> XPU.
Uses LLM knowledge instead of local YAML knowledge base.
"""

__version__ = "0.5.0"
__all__ = [
    "Config",
    "OptimizationResult",
    "OptimizationStage",
    "XForgePipeline",
    "get_config",
    "override_config",
]


# The prefix of this tool's environment variables before it was named X-Forge: one still set
# would be ignored, and what it set would silently fall back to a default.
OLD_ENV_PREFIX = "XE_FORGE_"


def load_env() -> None:
    """Load ``.env`` before anything reads the environment: the working directory's, then
    the checkout's; a variable already set wins. A framework's venv may lack python-dotenv.
    Exits when a variable still has the old prefix (:data:`OLD_ENV_PREFIX`)."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(find_dotenv(usecwd=True))
        load_dotenv()
    import os

    old = sorted(k for k in os.environ if k.startswith(OLD_ENV_PREFIX))
    if old:
        renamed = ", ".join(f"{k} -> X_FORGE_{k[len(OLD_ENV_PREFIX) :]}" for k in old)
        raise SystemExit(f"x-forge: environment variables were renamed: {renamed}")


# Lazy: the capture tools run inside a framework's venv and must not pull in the config stack.
_LAZY = {
    "Config": "x_forge.config",
    "get_config": "x_forge.config",
    "override_config": "x_forge.config",
    "OptimizationResult": "x_forge.models",
    "OptimizationStage": "x_forge.models",
    "XForgePipeline": "x_forge.pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
