"""
How accepted changes compose into a reported total: the headline number is measured,
never summed. A cumulative gain is a fresh end-to-end measurement of the full stack;
per-entry figures are kept but labelled unsummable, and the gap between the two is
reported as drift. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from xe_forge.orbit.models import Decision


class GainMethod(StrEnum):
    """How an entry's contribution to the chain was established.

    The value of naming these is that they are not equally trustworthy, and a report
    that renders them identically invites the reader to treat them as if they were.
    """

    # Re-measured end to end with this change and everything before it applied. The
    # only method that supports a cumulative claim.
    MEASURED = "measured"
    # The change was accepted on its own paired comparison, but the stack was never
    # re-measured with it in place. Its local delta is real; its contribution is not
    # established.
    LOCAL_ONLY = "local_only"
    # No usable figure at all — recorded for completeness so the entry cannot silently
    # vanish from the accounting.
    MISSING = "missing"


@dataclass
class StackEntry:
    """One accepted change, and what is actually known about its contribution."""

    label: str
    # The paired delta that got this change accepted. Real, and about this change
    # in isolation. Never summed with any other entry's.
    local_delta_percent: float
    decision: Decision = Decision.ACCEPT
    # End-to-end throughput measured with this entry and all prior entries applied.
    # None means nobody re-measured, which is a fact about the run, not about the entry.
    throughput_after: float | None = None
    unit: str = "tok/s"

    @property
    def gain_method(self) -> GainMethod:
        if self.throughput_after is not None:
            return GainMethod.MEASURED
        if self.local_delta_percent:
            return GainMethod.LOCAL_ONLY
        return GainMethod.MISSING


@dataclass
class CumulativeResult:
    """The validated total, the unsummable parts, and the gap between them."""

    baseline_throughput: float
    final_throughput: float | None
    entries: list[StackEntry] = field(default_factory=list)
    unit: str = "tok/s"

    @property
    def validated_gain_percent(self) -> float | None:
        """The headline. A measurement of the whole stack, or nothing at all.

        Returning None rather than a computed fallback is the point: a run that never
        re-measured its final stack has no cumulative result, and saying so is more
        useful than a plausible number nobody took.
        """
        if self.final_throughput is None or self.baseline_throughput <= 0:
            return None
        return (self.final_throughput / self.baseline_throughput - 1.0) * 100.0

    @property
    def naive_sum_percent(self) -> float:
        """What adding the per-entry percentages would have produced.

        Computed only so the report can show what it is *not* claiming. Never returned
        as the headline, and never used when the validated figure is missing.
        """
        return sum(e.local_delta_percent for e in self.entries)

    @property
    def compounded_percent(self) -> float:
        """The per-entry deltas compounded rather than added.

        Still not a valid cumulative gain — it corrects the arithmetic error but not
        the overlap or the drift — and shown for the same reason as the naive sum: to
        make visible how far a plausible-looking derivation lands from the measurement.
        """
        product = 1.0
        for entry in self.entries:
            product *= 1.0 + entry.local_delta_percent / 100.0
        return (product - 1.0) * 100.0

    @property
    def drift_percent(self) -> float | None:
        """Validated total minus what the parts claim, in percentage points.

        Negative means the stack delivers less than its entries promised — overlap or
        interference. Positive means it delivers more, which is not automatically good
        news either: it usually means something changed that nobody attributed.
        """
        validated = self.validated_gain_percent
        if validated is None:
            return None
        return validated - self.compounded_percent

    @property
    def chain_continuous(self) -> bool:
        """Whether every entry was re-measured, so drift can be attributed per step."""
        return bool(self.entries) and all(
            e.gain_method is GainMethod.MEASURED for e in self.entries
        )

    def format(self) -> str:
        lines = [
            f"{'STEP':<28} {'LOCAL Δ':>9} {'AFTER':>12}  METHOD",
            "-" * 76,
        ]
        for entry in self.entries:
            after = f"{entry.throughput_after:.1f}" if entry.throughput_after is not None else "—"
            lines.append(
                f"{entry.label[:28]:<28} {entry.local_delta_percent:>+8.2f}% "
                f"{after:>12}  {entry.gain_method.value}"
            )
        lines.append("-" * 76)

        validated = self.validated_gain_percent
        if validated is None:
            lines.append(
                "cumulative gain: NOT ESTABLISHED — the final stack was never measured end to end."
            )
            lines.append(
                "The per-step figures above are each real in isolation. Adding them "
                "would produce a number no measurement supports, so none is shown."
            )
            return "\n".join(lines)

        lines.append(
            f"baseline {self.baseline_throughput:.1f} {self.unit} -> "
            f"final {self.final_throughput:.1f} {self.unit}"
        )
        lines.append(f"CUMULATIVE GAIN (measured):   {validated:+.2f}%")
        lines.append(
            f"  per-step sum (not a result): {self.naive_sum_percent:+.2f}%  "
            f"compounded: {self.compounded_percent:+.2f}%"
        )
        drift = self.drift_percent
        if drift is not None:
            lines.append(f"  unattributed drift:          {drift:+.2f} points")
            if abs(drift) >= 5.0:
                lines.append(
                    "  The parts and the whole disagree by more than 5 points. That gap "
                    "is a finding: overlapping wins, interference, or a change nobody "
                    "attributed."
                )
        if not self.chain_continuous:
            lines.append(
                "  chain is discontinuous: at least one step was never re-measured, so "
                "the drift above cannot be attributed to individual steps."
            )
        return "\n".join(lines)


def accumulate(
    baseline_throughput: float,
    entries: list[StackEntry],
    final_throughput: float | None = None,
    unit: str = "tok/s",
) -> CumulativeResult:
    """Assemble the stack's accounting, taking the final measurement as authoritative.

    When `final_throughput` is not supplied, the last entry that was actually
    re-measured stands in for it — that is still a measurement of a real stack, just
    not of the final one. If no entry was ever re-measured there is no cumulative
    result, and the report says so rather than deriving one.
    """
    if final_throughput is None:
        measured = [e.throughput_after for e in entries if e.throughput_after is not None]
        final_throughput = measured[-1] if measured else None

    return CumulativeResult(
        baseline_throughput=baseline_throughput,
        final_throughput=final_throughput,
        entries=list(entries),
        unit=unit,
    )
