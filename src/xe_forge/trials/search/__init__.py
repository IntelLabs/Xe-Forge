"""Search strategy registry. Add new strategies by calling register_search()."""

from xe_forge.trials.search.base import TrialProposal, TrialSearch

_REGISTRY: dict[str, type[TrialSearch]] = {}


def register_search(name: str, cls: type[TrialSearch]) -> None:
    """Register a search strategy class under a string name."""
    _REGISTRY[name] = cls


def get_search(name: str) -> TrialSearch:
    """Look up and instantiate a search strategy by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown search strategy '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_searches() -> list[str]:
    """Names of all registered search strategies."""
    return sorted(_REGISTRY)


# Register built-in strategies. Imports at module-bottom so the base ABC is
# already available when the concrete classes import from it.
from xe_forge.trials.search.tree_walk import TreeWalkSearch  # noqa: E402
from xe_forge.trials.search.best_first import BestFirstSearch  # noqa: E402
from xe_forge.trials.search.beam import BeamSearch  # noqa: E402
from xe_forge.trials.search.mcts import MCTSSearch  # noqa: E402

register_search("tree_walk", TreeWalkSearch)
register_search("best_first", BestFirstSearch)
register_search("beam", BeamSearch)
register_search("mcts", MCTSSearch)

__all__ = [
    "TrialProposal",
    "TrialSearch",
    "register_search",
    "get_search",
    "list_searches",
]
