"""
The stall gate: an attempt identical to one already made does not run again.
Timeouts get a bounded retry allowance rather than counting as a real outcome.
Design rationale: docs/DESIGN.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

# Retries allowed after a timeout before the attempt counts as a repeat.
DEFAULT_TIMEOUT_RETRIES = 1


class Verdict(StrEnum):
    NOVEL = "novel"
    # Seen before with a real outcome.
    STALL = "stall"
    # Seen before, but only as a timeout, and the retry allowance is not exhausted.
    RETRY = "retry"


@dataclass(frozen=True)
class Attempt:
    """What was tried, in the terms that determine the outcome.

    `parameters` is part of identity; it is normalized through sorted JSON so dict
    ordering cannot make a repeat look novel.
    """

    action: str
    target: str = ""
    parameters: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.action,
                self.target,
                json.dumps(self.parameters, sort_keys=True, default=str),
            )
        )


@dataclass
class _Record:
    outcomes: int = 0
    timeouts: int = 0


@dataclass
class NoveltyLedger:
    """Remembers what has been attempted, so the loop cannot re-tread it."""

    timeout_retries: int = DEFAULT_TIMEOUT_RETRIES
    _seen: dict[str, _Record] = field(default_factory=dict)

    def classify(self, attempt: Attempt) -> tuple[Verdict, str]:
        """Decide whether this attempt may run, returning the verdict and its reason."""
        record = self._seen.get(attempt.key)
        if record is None:
            return Verdict.NOVEL, "not attempted before"

        if record.outcomes:
            return (
                Verdict.STALL,
                f"identical attempt already ran {record.outcomes}x with a real outcome; "
                f"repeating it spends budget to learn what is already known",
            )

        if record.timeouts <= self.timeout_retries:
            return (
                Verdict.RETRY,
                f"previous attempt timed out ({record.timeouts}x); a timeout describes "
                f"the machine rather than the attempt, so one retry is allowed",
            )

        return (
            Verdict.STALL,
            f"timed out {record.timeouts}x, past the retry allowance of "
            f"{self.timeout_retries}; treat 'does not finish' as the result",
        )

    def admits(self, attempt: Attempt) -> bool:
        return self.classify(attempt)[0] is not Verdict.STALL

    def record(self, attempt: Attempt, timed_out: bool = False) -> None:
        """Register that this attempt ran, so a later identical one is recognised."""
        record = self._seen.setdefault(attempt.key, _Record())
        if timed_out:
            record.timeouts += 1
        else:
            record.outcomes += 1

    @property
    def distinct_attempts(self) -> int:
        return len(self._seen)

    @property
    def total_attempts(self) -> int:
        return sum(r.outcomes + r.timeouts for r in self._seen.values())

    def format(self) -> str:
        if not self._seen:
            return "novelty ledger: nothing attempted yet"
        wasted = self.total_attempts - self.distinct_attempts
        lines = [
            f"novelty ledger: {self.distinct_attempts} distinct attempt(s), "
            f"{self.total_attempts} run(s)",
        ]
        if wasted:
            lines.append(
                f"  {wasted} run(s) were repeats — admitted as timeout retries, since a "
                f"repeat with a real outcome is refused."
            )
        return "\n".join(lines)
