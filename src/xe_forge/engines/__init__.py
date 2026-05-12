"""Engine layer: dispatches optimization to DSPy pipeline or Claude Code workspace."""

from xe_forge.engines.base import BaseEngine


def create_engine(config) -> BaseEngine:
    """Factory: create the engine specified by ``config.engine.engine``."""
    name = config.engine.engine.lower()
    if name == "claude":
        from xe_forge.engines.claude_engine import ClaudeEngine

        return ClaudeEngine(config)
    from xe_forge.engines.dspy_engine import DSPyEngine

    return DSPyEngine(config)


__all__ = ["BaseEngine", "create_engine"]
