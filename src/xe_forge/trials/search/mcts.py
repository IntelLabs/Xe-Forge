"""MCTS search skeleton (not yet implemented)."""

from typing import TYPE_CHECKING

from xe_forge.trials.search.base import TrialProposal, TrialSearch

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext


class MCTSSearch(TrialSearch):
    """UCB over the trial tree.

    Not implemented. The registry still resolves ``mcts`` so users get a
    clear error instead of a silent fallback.
    """

    def propose(self, state: dict, ctx: "TrialContext") -> TrialProposal:
        raise NotImplementedError(
            "MCTSSearch is a placeholder. Implement UCB selection in "
            "xe_forge/trials/search/mcts.py."
        )

    def should_stop(self, state: dict, ctx: "TrialContext") -> bool:
        return True
