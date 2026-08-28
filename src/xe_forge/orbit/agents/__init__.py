"""Repository agents: repository-level questions go through the `RepoAgent` protocol
with the provider selected by config, so Orbit never imports LLM tooling directly and
CPU-only CI can import these modules without an LLM client installed."""

from __future__ import annotations

import os

from xe_forge.orbit.agents.base import (
    AgentAnswer,
    AgentTask,
    BaseRepoAgent,
    RepoAgent,
    RepoAgentError,
)
from xe_forge.orbit.agents.claude_agent import ClaudeRepoAgent

_PROVIDERS: dict[str, type] = {
    "claude": ClaudeRepoAgent,
}

PROVIDER_ENV = "ORBIT_REPO_AGENT"


def available_providers() -> dict[str, type]:
    return dict(_PROVIDERS)


def get_agent(name: str | None = None, **kwargs) -> RepoAgent:
    """Build a repo agent by name, or from config, defaulting to Claude Code."""
    chosen = (name or os.environ.get(PROVIDER_ENV) or "claude").lower()
    if chosen not in _PROVIDERS:
        raise RepoAgentError(f"unknown repo agent {chosen!r}; available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[chosen](**kwargs)


def default_agent(**kwargs) -> RepoAgent | None:
    """The configured agent if it can actually run, otherwise None.

    Returning None rather than raising lets a caller fall back to a deterministic path
    and *say* that the agent was unavailable, instead of failing a resolution that had a
    perfectly good non-agent answer.
    """
    try:
        agent = get_agent(**kwargs)
    except RepoAgentError:
        return None
    return agent if agent.available() else None


__all__ = [
    "AgentAnswer",
    "AgentTask",
    "BaseRepoAgent",
    "ClaudeRepoAgent",
    "RepoAgent",
    "RepoAgentError",
    "available_providers",
    "default_agent",
    "get_agent",
]
