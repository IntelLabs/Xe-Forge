"""Beam search skeleton (not yet implemented)."""

from typing import TYPE_CHECKING

from xe_forge.trials.search.base import TrialProposal, TrialSearch

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext


class BeamSearch(TrialSearch):
    """Parallel beam over the top-K leaves.

    Not implemented. The registry still resolves ``beam`` so users get a
    clear error instead of a silent fallback.
    """

    def propose(self, state: dict, ctx: "TrialContext") -> TrialProposal:
        raise NotImplementedError(
            "BeamSearch is a placeholder. Implement expand-K-leaves logic "
            "in xe_forge/trials/search/beam.py."
        )

    def should_stop(self, state: dict, ctx: "TrialContext") -> bool:
        return True
