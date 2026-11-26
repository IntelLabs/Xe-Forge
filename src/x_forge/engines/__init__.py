"""Engine layer: dispatches optimization to DSPy pipeline or Claude Code workspace."""

from x_forge.engines.base import BaseEngine


def create_engine(config) -> BaseEngine:
    """Factory: create the engine specified by ``config.engine.engine``."""
    name = config.engine.engine.lower()
    if name in ("claude", "codex"):  # the same workspace, driven by either agent CLI
        from x_forge.engines.claude_engine import ClaudeEngine

        return ClaudeEngine(config)
    if name == "dspy-agent":
        from x_forge.engines.agent_engine import AgentEngine

        return AgentEngine(config)
    from x_forge.engines.dspy_engine import DSPyEngine

    return DSPyEngine(config)


__all__ = ["BaseEngine", "create_engine"]
