"""Writer registry.

Writers need agents injected (analyzer, optimizer, etc.), so the registry
stores classes, and the runner calls ``get_writer_cls(name)`` and constructs
the writer with the deps it already has. Writers that need no deps can still
be constructed via ``get_writer(name)``.
"""

from xe_forge.trials.writers.base import TrialWriter

_REGISTRY: dict[str, type[TrialWriter]] = {}


def register_writer(name: str, cls: type[TrialWriter]) -> None:
    """Register a writer class under a string name."""
    _REGISTRY[name] = cls


def get_writer_cls(name: str) -> type[TrialWriter]:
    """Return the writer class registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown writer '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def get_writer(name: str, **kwargs) -> TrialWriter:
    """Instantiate a writer (pass deps via kwargs)."""
    return get_writer_cls(name)(**kwargs)


def list_writers() -> list[str]:
    """Names of all registered writers."""
    return sorted(_REGISTRY)


# Register built-ins at module bottom so concrete classes can import from
# the base without a cycle.
from xe_forge.trials.writers.stage_sequence import StageSequenceWriter  # noqa: E402

register_writer("stage_sequence", StageSequenceWriter)


__all__ = [
    "TrialWriter",
    "register_writer",
    "get_writer_cls",
    "get_writer",
    "list_writers",
]
