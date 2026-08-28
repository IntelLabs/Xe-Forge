"""
Weighted multi-variant objective (plan §9.1).

`--variant` selects one benchmark configuration; a real workload runs a distribution
of them. This module scores a candidate across a whole variant family —
`score(C) = Σ wᵢ · latency(C, variantᵢ)` — with the hard constraint §9.1 states:
**no variant may regress**, whatever the weighted total says. A candidate tuned for
the dominant shape that collapses on the tail distribution wins the microbenchmark
and loses end to end, and the per-variant table below is what catches it.

Two rules carried over from the measurement methodology:

* The verdict is per-variant, never a single number (§14.3). `format()` renders the
  table; `weighted_speedup` is the headline only alongside it.
* A variant that fails to run is a failure, not a gap. A family member the candidate
  cannot execute is worse than one it slows down, and folding it into "no data"
  would report a broken kernel as a partial win.

The executor is duck-typed: anything with `compare_kernels(original_code,
optimized_code, kernel_name=..., input_shapes=..., flop=..., dtype=...,
init_args=..., input_dtypes=...)` returning an object with `original_time_us`,
`optimized_time_us`, `speedup` and (optionally) `optimized_correct`. That keeps this
module importable and testable without ai_bench.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A regression smaller than this is indistinguishable from run-to-run noise at the
# kernel level; beyond it, the no-regression constraint trips. Callers with a
# measured noise floor should pass their own.
DEFAULT_REGRESSION_TOLERANCE_PERCENT = 1.0


def family_base(variant_type: str) -> str:
    """The family a variant key belongs to: `bench-gpu-3` → `bench-gpu`."""
    return re.sub(r"-\d+$", "", variant_type)


@dataclass
class VariantOutcome:
    """One family member's measured result."""

    variant: str
    index: int
    weight: float  # normalized share of the objective
    declared_weight: float | None
    speedup: float
    original_us: float
    optimized_us: float
    correct: bool
    error: str = ""

    @property
    def ran(self) -> bool:
        return not self.error


@dataclass
class WeightedComparison:
    """The full §9.1 verdict: a table, a weighted headline, and a hard constraint."""

    family: str
    outcomes: list[VariantOutcome] = field(default_factory=list)
    weighted_speedup: float | None = None
    regressions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    accepted: bool = False
    reason: str = ""

    def format(self) -> str:
        lines = [
            f"weighted objective over {self.family!r} ({len(self.outcomes)} variant(s)):",
            f"{'VARIANT':<16} {'WEIGHT':>7} {'ORIG us':>10} {'OPT us':>10} {'SPEEDUP':>8}  NOTE",
            "-" * 72,
        ]
        for o in self.outcomes:
            if o.ran:
                note = ""
                if o.variant in self.regressions:
                    note = "REGRESSION"
                elif not o.correct:
                    note = "INCORRECT"
                lines.append(
                    f"{o.variant:<16} {o.weight:>7.3f} {o.original_us:>10.1f} "
                    f"{o.optimized_us:>10.1f} {o.speedup:>7.2f}x  {note}"
                )
            else:
                lines.append(
                    f"{o.variant:<16} {o.weight:>7.3f} {'-':>10} {'-':>10} {'-':>8}  {o.error}"
                )
        lines.append("-" * 72)
        headline = f"{self.weighted_speedup:.2f}x" if self.weighted_speedup else "not established"
        lines.append(f"weighted speedup: {headline}")
        lines.append(f"{'ACCEPTED' if self.accepted else 'REJECTED'}: {self.reason}")
        return "\n".join(lines)

    def summary(self) -> dict:
        """A JSON-serializable record for OptimizationResult / run artifacts."""
        return {
            "family": self.family,
            "weighted_speedup": self.weighted_speedup,
            "accepted": self.accepted,
            "reason": self.reason,
            "regressions": self.regressions,
            "failures": self.failures,
            "variants": [
                {
                    "variant": o.variant,
                    "weight": o.weight,
                    "declared_weight": o.declared_weight,
                    "speedup": o.speedup if o.ran else None,
                    "original_us": o.original_us if o.ran else None,
                    "optimized_us": o.optimized_us if o.ran else None,
                    "correct": o.correct,
                    "error": o.error,
                }
                for o in self.outcomes
            ],
        }


def _normalized_weights(declared: list[float | None], family: str) -> list[float]:
    """Turn declared weights into shares, refusing the ambiguous mix.

    All declared → normalized to sum 1. None declared → equal shares. A mix would
    force a guess about what the author meant for the unweighted entries, and every
    guess silently reweights the objective — so it is an error, stated as one.
    """
    present = [w for w in declared if w is not None]
    if not present:
        return [1.0 / len(declared)] * len(declared)
    if len(present) != len(declared):
        raise ValueError(
            f"family {family!r} mixes weighted and unweighted variants; either every "
            f"variant declares `weight:` or none does (§9.1)"
        )
    total = sum(present)
    if total <= 0:
        return [1.0 / len(declared)] * len(declared)
    return [w / total for w in present]


def compare_weighted(
    executor,
    spec,
    original_code: str,
    optimized_code: str,
    kernel_name: str | None = None,
    family: str = "bench-gpu",
    regression_tolerance_percent: float = DEFAULT_REGRESSION_TOLERANCE_PERCENT,
    required_speedup: float | None = None,
) -> WeightedComparison:
    """Score a candidate across a variant family (§9.1).

    Accepts only when every variant ran, none regressed beyond the tolerance, none
    was incorrect, and the weighted speedup clears `required_speedup` (default: any
    improvement). `spec` is a `KernelSpec`; `family` is the base key whose numbered
    siblings carry the shape distribution (§8).
    """
    result = WeightedComparison(family=family)

    triples = spec.weighted_family(family)
    if not triples:
        result.reason = f"spec has no variants in family {family!r}"
        return result

    weights = _normalized_weights([v.weight for _, _, v in triples], family)
    floor = 1.0 - regression_tolerance_percent / 100.0

    weighted_orig = 0.0
    weighted_opt = 0.0
    measured_weight = 0.0

    for (key, index, variant), share in zip(triples, weights, strict=True):
        comparison = executor.compare_kernels(
            original_code,
            optimized_code,
            kernel_name=kernel_name,
            input_shapes=spec.get_input_shapes(key, index),
            flop=spec.get_flop(key, index),
            dtype=spec.get_dtype(key, index),
            init_args=spec.get_init_args(key, index),
            input_dtypes=spec.get_input_dtypes(key, index),
        )

        orig_us = float(getattr(comparison, "original_time_us", 0.0) or 0.0)
        opt_us = float(getattr(comparison, "optimized_time_us", 0.0) or 0.0)
        speedup = float(getattr(comparison, "speedup", 0.0) or 0.0)
        correct = bool(getattr(comparison, "optimized_correct", True))
        ran = speedup > 0 and orig_us not in (0.0, float("inf")) and opt_us != float("inf")

        outcome = VariantOutcome(
            variant=key,
            index=index,
            weight=share,
            declared_weight=variant.weight,
            speedup=speedup,
            original_us=orig_us,
            optimized_us=opt_us,
            correct=correct,
            error="" if ran else str(getattr(comparison, "feedback_message", "") or "failed"),
        )
        result.outcomes.append(outcome)

        if not ran:
            result.failures.append(key)
            continue
        if speedup < floor:
            result.regressions.append(key)

        weighted_orig += share * orig_us
        weighted_opt += share * opt_us
        measured_weight += share

    if weighted_opt > 0 and measured_weight > 0:
        result.weighted_speedup = weighted_orig / weighted_opt

    incorrect = [o.variant for o in result.outcomes if o.ran and not o.correct]
    threshold = required_speedup if required_speedup is not None else 1.0

    if result.failures:
        result.reason = f"variant(s) failed to run: {', '.join(result.failures)}"
    elif incorrect:
        result.reason = f"variant(s) numerically incorrect: {', '.join(incorrect)}"
    elif result.regressions:
        # The hard constraint. A weighted win that regresses a family member is a
        # trade, and a trade must be surfaced as one, never accepted as a win (§14.3).
        result.reason = (
            f"weighted speedup {result.weighted_speedup:.2f}x, but variant(s) regressed "
            f"beyond {regression_tolerance_percent:g}%: {', '.join(result.regressions)}"
        )
    elif result.weighted_speedup is None:
        result.reason = "no variant produced a usable measurement"
    elif result.weighted_speedup < threshold:
        result.reason = (
            f"weighted speedup {result.weighted_speedup:.2f}x is below the required "
            f"{threshold:.2f}x"
        )
    else:
        result.accepted = True
        result.reason = (
            f"weighted speedup {result.weighted_speedup:.2f}x across "
            f"{len(result.outcomes)} variant(s), no regressions"
        )
    return result
