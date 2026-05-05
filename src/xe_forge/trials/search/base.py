"""Search ABC + TrialProposal dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext


@dataclass
class TrialProposal:
    """The search's decision for the next trial."""

    parent_id: str      # trial ID to branch from (e.g. "t0", "t3")
    strategy: str       # human-readable hint, persisted in state.json
    hint_tag: str       # machine-readable: first_try, extend, branch_back,
                        #                  try_harder, fix_on_same_branch


class TrialSearch(ABC):
    """Pluggable tree-search policy.

    ``propose`` picks which trial to branch from next and with what strategy.
    ``should_stop`` decides when the loop ends. Both see the full state dict
    (from ``store.get_state``) plus the shared ``TrialContext``.
    """

    @abstractmethod
    def propose(self, state: dict, ctx: "TrialContext") -> TrialProposal:
        """Return the parent and strategy for the next trial."""

    @abstractmethod
    def should_stop(self, state: dict, ctx: "TrialContext") -> bool:
        """Return True when the loop should terminate."""

    # Optional: searches can override this to signal a plateau to the runner,
    # which triggers a VTune profile. Default: no plateau detection.
    def is_plateau(self, state: dict, ctx: "TrialContext") -> bool:
        return False
