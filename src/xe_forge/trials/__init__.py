"""Tree-structured trial exploration engine for Xe-Forge.

Replaces the flat best_k loop with a pluggable tree search. Each trial is a
node in a tree rooted at t0; a TrialSearch picks the parent to branch from,
a TrialWriter produces the child kernel, and the runner enforces
single-XPU serialization.
"""

from xe_forge.trials.context import TrialContext
from xe_forge.trials.runner import TrialRunner, TrialRunResult
from xe_forge.trials.search import get_search, register_search
from xe_forge.trials.writers import get_writer, get_writer_cls, register_writer

__all__ = [
    "TrialContext",
    "TrialRunner",
    "TrialRunResult",
    "get_search",
    "register_search",
    "get_writer",
    "get_writer_cls",
    "register_writer",
]
