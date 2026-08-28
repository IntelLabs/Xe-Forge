"""
Measurement statistics and the accept/reject arithmetic. Samples become intervals,
intervals become decisions; `INCONCLUSIVE` is a real outcome, not a soft `REJECT`.
Deliberately stdlib-only so it imports fast with no scientific stack present.
"""

from __future__ import annotations

import math
from statistics import mean as _mean
from statistics import stdev as _stdev

from xe_forge.orbit.models import Decision, MetricEstimate

# Two-sided 95% Student's t critical values by df; the table keeps scipy out of the
# dependency set, and beyond df=30 the normal approximation is within ~1%.
_T95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_Z95 = 1.960
# One-sided z for 80% power, used only in the minimum-detectable-effect formula.
_Z80 = 0.842

# Coefficient of variation above which a run's clocks are considered unstable.
DEFAULT_CLOCK_CV_THRESHOLD = 0.05


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value for `df` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df in _T95:
        return _T95[df]
    return _Z95


def estimate(samples: list[float], unit: str = "") -> MetricEstimate:
    """Turn raw samples into a mean with a 95% confidence interval.

    A single sample yields a degenerate interval (low == high == mean) and n=1, which
    the decision rule then refuses to accept.
    """
    if not samples:
        raise ValueError("estimate() requires at least one sample")
    n = len(samples)
    m = _mean(samples)
    if n == 1:
        return MetricEstimate(
            mean=m, stdev=0.0, n=1, ci95_low=m, ci95_high=m, samples=list(samples), unit=unit
        )
    sd = _stdev(samples)
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    return MetricEstimate(
        mean=m,
        stdev=sd,
        n=n,
        ci95_low=m - half,
        ci95_high=m + half,
        samples=list(samples),
        unit=unit,
    )


def minimum_detectable_effect(samples: list[float], n_planned: int | None = None) -> float:
    """Smallest relative effect (as a percent of the mean) this setup can resolve.

    Standard two-sample formula at alpha=0.05 / power=0.80. An Amdahl ceiling below
    this number means optimizing the kernel cannot produce a resolvable result.
    """
    if len(samples) < 2:
        return float("inf")
    m = _mean(samples)
    if m == 0:
        return float("inf")
    sd = _stdev(samples)
    n = n_planned or len(samples)
    if n < 2:
        return float("inf")
    # Standard error of a difference of two means, each estimated from n samples.
    se_diff = sd * math.sqrt(2.0 / n)
    mde_abs = (t_critical_95(2 * n - 2) + _Z80) * se_diff
    return abs(mde_abs / m) * 100.0


def clocks_stable(
    clock_samples: list[float], threshold: float = DEFAULT_CLOCK_CV_THRESHOLD
) -> bool:
    """True when observed clock variation is low enough for a run to be comparable.

    An empty sample list means clocks could not be read at all, which is not the same
    claim as instability.
    """
    if len(clock_samples) < 2:
        return True
    m = _mean(clock_samples)
    if m <= 0:
        return True
    return (_stdev(clock_samples) / m) <= threshold


def _welch_interval(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """(difference of means b-a, ci_low, ci_high) for two independent samples."""
    na, nb = len(a), len(b)
    ma, mb = _mean(a), _mean(b)
    va = _stdev(a) ** 2 if na > 1 else 0.0
    vb = _stdev(b) ** 2 if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb)
    diff = mb - ma
    if se == 0:
        return diff, diff, diff
    # Welch-Satterthwaite degrees of freedom.
    num = (va / na + vb / nb) ** 2
    den = 0.0
    if na > 1:
        den += (va / na) ** 2 / (na - 1)
    if nb > 1:
        den += (vb / nb) ** 2 / (nb - 1)
    df = int(num / den) if den > 0 else 1
    half = t_critical_95(df) * se
    return diff, diff - half, diff + half


def _paired_interval(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """(mean paired difference b-a, ci_low, ci_high) for interleaved A/B runs."""
    # strict=True is the invariant: every baseline run must have its interleaved pair.
    diffs = [bi - ai for ai, bi in zip(a, b, strict=True)]
    n = len(diffs)
    md = _mean(diffs)
    if n < 2:
        return md, md, md
    sd = _stdev(diffs)
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    return md, md - half, md + half


def compare(
    baseline: list[float],
    candidate: list[float],
    *,
    lower_is_better: bool = True,
    paired: bool | None = None,
    min_repetitions: int = 5,
    clock_samples: list[float] | None = None,
    mde_percent: float | None = None,
) -> tuple[Decision, dict[str, float | str]]:
    """Compare candidate against baseline and return a decision plus its evidence.

    `paired` defaults to True when the sample lists are the same length (what
    interleaved A,B,A,B execution produces); pairing removes drift both arms shared.
    Returns the decision and a detail dict with the improvement percentage, its
    confidence interval, and the reason.
    """
    detail: dict[str, float | str] = {}

    if not baseline or not candidate:
        return Decision.INVALID, {"reason": "empty sample set"}

    if clock_samples is not None and not clocks_stable(clock_samples):
        return Decision.INVALID, {"reason": "clock variance above threshold"}

    if len(baseline) < min_repetitions or len(candidate) < min_repetitions:
        detail["reason"] = (
            f"insufficient repetitions: {len(baseline)} baseline / {len(candidate)} "
            f"candidate, need {min_repetitions}"
        )
        return Decision.INVALID, detail

    if paired is None:
        paired = len(baseline) == len(candidate)

    if paired and len(baseline) == len(candidate):
        diff, lo, hi = _paired_interval(baseline, candidate)
        method = "paired"
    else:
        diff, lo, hi = _welch_interval(baseline, candidate)
        method = "welch"

    base_mean = _mean(baseline)
    if base_mean == 0:
        return Decision.INVALID, {"reason": "baseline mean is zero"}

    # Express everything as percent improvement, sign-corrected for the metric's
    # direction, so a positive number always means "better".
    sign = -1.0 if lower_is_better else 1.0
    improvement = sign * diff / base_mean * 100.0
    imp_lo = sign * lo / base_mean * 100.0
    imp_hi = sign * hi / base_mean * 100.0
    if imp_lo > imp_hi:
        imp_lo, imp_hi = imp_hi, imp_lo

    detail.update(
        {
            "method": method,
            "improvement_percent": improvement,
            "ci95_low": imp_lo,
            "ci95_high": imp_hi,
        }
    )

    effective_mde = mde_percent if mde_percent is not None else minimum_detectable_effect(baseline)
    detail["minimum_detectable_effect"] = effective_mde

    if imp_lo <= 0.0 <= imp_hi:
        detail["reason"] = (
            f"95% CI [{imp_lo:.2f}%, {imp_hi:.2f}%] straddles zero (MDE {effective_mde:.2f}%)"
        )
        return Decision.INCONCLUSIVE, detail

    if imp_lo > 0.0:
        detail["reason"] = f"95% CI [{imp_lo:.2f}%, {imp_hi:.2f}%] excludes zero, positive"
        return Decision.ACCEPT, detail

    detail["reason"] = f"95% CI [{imp_lo:.2f}%, {imp_hi:.2f}%] excludes zero, negative"
    return Decision.REJECT, detail


def amdahl_ceiling(share: float, speedup: float, gpu_busy_fraction: float) -> float:
    """Maximum end-to-end gain (percent) from speeding one kernel up.

        max_e2e_gain(k, s) = share(k) * (1 - 1/s) * gpu_busy_fraction

    Returned as a percentage so it compares directly against the MDE.
    """
    if speedup <= 0:
        return 0.0
    if speedup <= 1.0:
        return 0.0
    return share * (1.0 - 1.0 / speedup) * gpu_busy_fraction * 100.0
