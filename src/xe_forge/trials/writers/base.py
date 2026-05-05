"""Writer ABC."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext
    from xe_forge.trials.search import TrialProposal


class TrialWriter(ABC):
    """Pluggable kernel-writer.

    Given a parent kernel's source code, a proposal from the search, the
    shared ``TrialContext``, and optionally a VTune report, return the source
    code of the next trial kernel.
    """

    @abstractmethod
    def write(
        self,
        parent_code: str,
        proposal: "TrialProposal",
        ctx: "TrialContext",
        vtune_report: str = "",
    ) -> str:
        """Return the new Triton kernel source."""
