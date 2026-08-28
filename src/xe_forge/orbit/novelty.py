"""
The stall gate: repeating an attempt is not progress (plan §20.4).

A loop that can retry will retry, and the cheapest thing for it to retry is whatever it
just did. Left alone that produces a run which looks busy — attempts logged, time spent,
budget consumed — and ends where it started, with the failure repeated N times in the
report instead of once.

The rule is one line: **an attempt identical to one already made does not get to run
again.** Identity is the tuple that determines the outcome — what was attempted, on what
target, with what parameters — so a genuinely different attempt is admitted and only a
literal repeat is refused.

Two distinctions decide whether this helps or gets in the way:

* **A repeat is a stall; a novel attempt is progress even if it also fails.** Failing
  differently is how a search moves. Only sameness is the problem.
* **A timeout is not a repeat.** The same attempt that timed out may succeed with more
  time or a warmer cache, because a timeout says something about the machine rather than
  about the attempt. Retrying it once is legitimate; retrying it forever is not, so
  timeouts get a bounded allowance rather than an exemption.

Convergent design: AMD's Hyperloom carries a "novelty-ledger stall gate" over the same
tuple — component, ref, GPU arch, build command — with the same timeout carve-out,
reverting on a repeat so the loop "keeps making forward progress rather than looping on
an identical failing build". Arriving at the same rule from a different loop is a
reasonable sign it is load-bearing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

# How many times one attempt may be retried after a timeout before it counts as a
# repeat. One retry distinguishes "the machine was busy" from "this does not finish";
# a second tells you nothing the first did not.
DEFAULT_TIMEOUT_RETRIES = 1


class Verdict(StrEnum):
    NOVEL = "novel"
    # Seen before with a real outcome. Running it again would consume budget to learn
    # something already known.
    STALL = "stall"
    # Seen before, but only as a timeout, and the retry allowance is not exhausted.
    RETRY = "retry"


@dataclass(frozen=True)
class Attempt:
    """What was tried, in the terms that determine the outcome.

    `parameters` is part of identity because the same action with different parameters
    is a different experiment. It is normalized through sorted JSON so that dict
    ordering — which says nothing about the attempt — cannot make a repeat look novel.
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
        """Decide whether this attempt may run, and say why in the same breath.

        The reason is returned rather than logged because a refusal the caller cannot
        explain to a user is indistinguishable from a bug.
        """
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
