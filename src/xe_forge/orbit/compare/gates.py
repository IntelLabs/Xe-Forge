"""
The L0-L5 correctness ladder (ordered, short-circuiting: a failed gate blocks the
rest) and matrix acceptance (a weighted win with no per-profile regression).
Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from xe_forge.orbit import stats
from xe_forge.orbit.models import Decision, WorkloadMatrix


class Gate(StrEnum):
    L0 = "L0"
    L0B = "L0b"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


GATE_DESCRIPTION = {
    Gate.L0: "build / import / registration succeeds",
    Gate.L0B: "extraction verified: right kernel, right specialization",
    Gate.L1: "kernel correctness vs captured reference (tightened tolerance)",
    Gate.L2: "weighted kernel latency improves, no variant regresses",
    Gate.L3: "model-level numerical gate",
    Gate.L4: "end-to-end performance with a confidence interval",
    Gate.L5: "re-profile confirms the kernel actually changed",
}

# Gates that may be skipped when the information to run them is genuinely absent (no
# model available for L3, for instance). L0b and L5 are NOT here on purpose.
SKIPPABLE = {Gate.L3}


@dataclass
class GateResult:
    gate: Gate
    passed: bool
    detail: str = ""
    skipped: bool = False

    @property
    def blocking(self) -> bool:
        return not self.passed and not self.skipped


@dataclass
class GateLadder:
    """Ordered gate results for one candidate."""

    kernel_id: str
    results: list[GateResult] = field(default_factory=list)

    def record(self, gate: Gate, passed: bool, detail: str = "", skipped: bool = False) -> None:
        self.results.append(GateResult(gate, passed, detail, skipped))

    @property
    def passed(self) -> bool:
        return bool(self.results) and not any(r.blocking for r in self.results)

    @property
    def failed_at(self) -> Gate | None:
        for result in self.results:
            if result.blocking:
                return result.gate
        return None

    def format(self) -> str:
        lines = [f"correctness ladder for {self.kernel_id}", "-" * 68]
        for result in self.results:
            mark = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
            lines.append(f"  [{mark}] {result.gate.value:<4} {GATE_DESCRIPTION[result.gate]}")
            if result.detail:
                lines.append(f"          {result.detail}")
        lines.append("-" * 68)
        failed = self.failed_at
        lines.append("RESULT: PASS" if self.passed else f"RESULT: FAIL — blocked at {failed.value}")
        return "\n".join(lines)


def run_ladder(
    kernel_id: str,
    *,
    build_ok: bool,
    build_detail: str = "",
    extraction_verified: bool,
    extraction_detail: str = "",
    correctness_ok: bool | None = None,
    correctness_detail: str = "",
    kernel_samples: tuple[list[float], list[float]] | None = None,
    model_gate_ok: bool | None = None,
    model_gate_detail: str = "",
    e2e_samples: tuple[list[float], list[float]] | None = None,
    reprofile_changed: bool | None = None,
    reprofile_detail: str = "",
    min_repetitions: int = 5,
) -> GateLadder:
    """Run the ladder in order, stopping at the first blocking failure.

    Short-circuiting is the point: an incorrect candidate must never reach L4, because
    a timing number for a wrong kernel is worse than no number at all — it looks like
    evidence.
    """
    ladder = GateLadder(kernel_id=kernel_id)

    ladder.record(
        Gate.L0, build_ok, build_detail or ("built and registered" if build_ok else "build failed")
    )
    if not build_ok:
        return ladder

    ladder.record(
        Gate.L0B,
        extraction_verified,
        extraction_detail
        or (
            "bundle proven to be the kernel that ran"
            if extraction_verified
            else "bundle unverified: this may be a different specialization of the right kernel"
        ),
    )
    if not extraction_verified:
        return ladder

    if correctness_ok is None:
        ladder.record(Gate.L1, False, "no correctness result supplied; refusing to assume")
        return ladder
    ladder.record(Gate.L1, correctness_ok, correctness_detail)
    if not correctness_ok:
        return ladder

    if kernel_samples is not None:
        baseline, candidate = kernel_samples
        decision, detail = stats.compare(baseline, candidate, min_repetitions=min_repetitions)
        improved = decision is Decision.ACCEPT
        ladder.record(Gate.L2, improved, str(detail.get("reason", "")))
        if not improved:
            return ladder
    else:
        ladder.record(Gate.L2, False, "no kernel-level samples supplied")
        return ladder

    if model_gate_ok is None:
        ladder.record(
            Gate.L3,
            True,
            model_gate_detail or "no model available for a numerical gate; not run",
            skipped=True,
        )
    else:
        ladder.record(Gate.L3, model_gate_ok, model_gate_detail)
        if not model_gate_ok:
            return ladder

    if e2e_samples is not None:
        baseline, candidate = e2e_samples
        decision, detail = stats.compare(baseline, candidate, min_repetitions=min_repetitions)
        ladder.record(
            Gate.L4,
            decision is Decision.ACCEPT,
            f"{decision.value}: {detail.get('reason', '')}",
        )
        if decision is not Decision.ACCEPT:
            return ladder
    else:
        ladder.record(Gate.L4, False, "no end-to-end samples supplied")
        return ladder

    if reprofile_changed is None:
        # L5 is not skippable: without it an end-to-end gain cannot be attributed to
        # this change rather than to something else that moved.
        ladder.record(
            Gate.L5,
            False,
            "no re-profile performed; an end-to-end gain cannot be attributed to this change",
        )
        return ladder
    ladder.record(Gate.L5, reprofile_changed, reprofile_detail)
    return ladder


# ---------------------------------------------------------------------------
# Matrix acceptance
# ---------------------------------------------------------------------------


@dataclass
class ProfileOutcome:
    profile_id: str
    decision: Decision
    improvement_percent: float
    ci95_low: float
    ci95_high: float
    weight: float
    reason: str = ""


@dataclass
class MatrixDecision:
    """Per-profile results plus the overall verdict. Never a single number."""

    decision: Decision
    weighted_improvement: float
    outcomes: list[ProfileOutcome] = field(default_factory=list)
    reason: str = ""

    @property
    def regressions(self) -> list[ProfileOutcome]:
        return [o for o in self.outcomes if o.decision is Decision.REJECT]

    def format(self) -> str:
        lines = [
            f"{'PROFILE':<24} {'WEIGHT':>7} {'DELTA':>9} {'95% CI':>20}  DECISION",
            "-" * 78,
        ]
        for outcome in self.outcomes:
            interval = f"[{outcome.ci95_low:+.2f}, {outcome.ci95_high:+.2f}]"
            lines.append(
                f"{outcome.profile_id:<24} {outcome.weight:>7.2f} "
                f"{outcome.improvement_percent:>8.2f}% {interval:>20}  {outcome.decision.value}"
            )
        lines.append("-" * 78)
        lines.append(
            f"weighted: {self.weighted_improvement:+.2f}%   verdict: {self.decision.value}"
        )
        if self.reason:
            lines.append(self.reason)
        return "\n".join(lines)


def decide_matrix(
    matrix: WorkloadMatrix,
    samples: dict[str, tuple[list[float], list[float]]],
    *,
    regression_threshold_percent: float = 2.0,
    min_repetitions: int = 5,
) -> MatrixDecision:
    """Accept only on a weighted win with no per-profile regression.

    `samples` maps profile id -> (baseline samples, candidate samples). A profile that
    regresses beyond `regression_threshold_percent` rejects the whole candidate even
    when the weighted average is positive: a trade must be surfaced as a trade.
    """
    weights = matrix.normalized_weights()
    outcomes: list[ProfileOutcome] = []

    for profile in matrix.profiles:
        pair = samples.get(profile.id)
        weight = weights.get(profile.id, 0.0)
        if pair is None:
            outcomes.append(
                ProfileOutcome(
                    profile_id=profile.id,
                    decision=Decision.INVALID,
                    improvement_percent=0.0,
                    ci95_low=0.0,
                    ci95_high=0.0,
                    weight=weight,
                    reason="no samples for this profile",
                )
            )
            continue

        baseline, candidate = pair
        decision, detail = stats.compare(baseline, candidate, min_repetitions=min_repetitions)
        outcomes.append(
            ProfileOutcome(
                profile_id=profile.id,
                decision=decision,
                improvement_percent=float(detail.get("improvement_percent", 0.0)),
                ci95_low=float(detail.get("ci95_low", 0.0)),
                ci95_high=float(detail.get("ci95_high", 0.0)),
                weight=weight,
                reason=str(detail.get("reason", "")),
            )
        )

    weighted = sum(o.improvement_percent * o.weight for o in outcomes)

    invalid = [o for o in outcomes if o.decision is Decision.INVALID]
    if invalid:
        return MatrixDecision(
            decision=Decision.INVALID,
            weighted_improvement=weighted,
            outcomes=outcomes,
            reason=f"{len(invalid)} profile(s) could not be measured: "
            + ", ".join(o.profile_id for o in invalid),
        )

    # A regression anywhere beyond the declared threshold rejects, regardless of the
    # weighted average. This is the "wins decode, loses prefill" case.
    regressions = [
        o for o in outcomes if o.improvement_percent < -abs(regression_threshold_percent)
    ]
    if regressions:
        detail = ", ".join(f"{o.profile_id} {o.improvement_percent:+.2f}%" for o in regressions)
        return MatrixDecision(
            decision=Decision.REJECT,
            weighted_improvement=weighted,
            outcomes=outcomes,
            reason=(
                f"per-profile regression beyond {regression_threshold_percent:.1f}%: {detail}. "
                f"This is a trade, not an improvement — report it as one."
            ),
        )

    if all(o.decision is Decision.INCONCLUSIVE for o in outcomes):
        return MatrixDecision(
            decision=Decision.INCONCLUSIVE,
            weighted_improvement=weighted,
            outcomes=outcomes,
            reason="no profile resolved a difference; the workload cannot measure this change",
        )

    if weighted > 0 and any(o.decision is Decision.ACCEPT for o in outcomes):
        return MatrixDecision(
            decision=Decision.ACCEPT,
            weighted_improvement=weighted,
            outcomes=outcomes,
            reason=f"weighted win of {weighted:+.2f}% with no profile regressing",
        )

    return MatrixDecision(
        decision=Decision.INCONCLUSIVE,
        weighted_improvement=weighted,
        outcomes=outcomes,
        reason="weighted improvement did not clear zero across the matrix",
    )
