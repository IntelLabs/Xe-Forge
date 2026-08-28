"""
Correctness for a kernel that never left the framework (plan §19.6, §19.7).

§19's ladder assumes an extracted kernel with a captured reference: L1 compares the
bundle's output against what the workload produced. That works at E1/E2 and not at E3,
where the framework drives dispatch and there is no standalone artifact to diff. E3 is
also the level most kernels reach, so the ladder's most-used rung had no implementation.

AMD's Hyperloom solves the same problem for a system that only ever works in place, and
its answer transfers directly because an in-situ kernel is reachable the ordinary way —
you import it. Their check is a small script that imports the framework's kernel,
computes a reference in higher precision by hand, and compares:

    from sglang.srt.layers.layernorm import RMSNorm   # the real kernel
    reference = xf32 * rsqrt * weight                 # hand-written, fp32
    rel = (diff / denom).max(dim=-1).values
    accuracy = (rel < rtol).sum() / rel.numel()

Two details there are the whole design, and both are easy to get wrong:

* **The metric is the fraction of rows within tolerance, not `allclose`.** `allclose` is
  a max over every element, so one bad value fails an otherwise correct kernel — and
  on a reduction over thousands of elements in bf16, one bad value is normal. A plain
  mean error has the opposite failure: a handful of catastrophically wrong elements
  vanish into an average that still looks small. Taking the max *within* a row and then
  the fraction of rows that pass is the middle: tolerant of scattered noise, and
  impossible to pass while being broadly wrong.
* **The reference is computed in higher precision than the kernel.** A bf16 reference
  compared against a bf16 kernel agrees on its own rounding error. The reference exists
  to be more accurate than the thing under test, or it is not a reference.

`denom` is clamped because relative error against a near-zero reference is unbounded and
says nothing about the kernel.

The second layer is separate and is not optional: a kernel can be numerically fine and
still leave the served model broken, so §19's L3 needs a task score too (§19.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Relative tolerance for a single element. Matches Xe-Forge's existing kernel rtol.
DEFAULT_RTOL = 1e-2

# Reference magnitudes below this are treated as this, because relative error against a
# near-zero value is unbounded and is a property of the reference, not the kernel.
DEFAULT_FLOOR = 1e-3

# Fraction of rows that must pass before a kernel is considered numerically correct.
DEFAULT_PASS_FRACTION = 0.99


@dataclass
class AccuracyResult:
    """How much of the output matched a higher-precision reference."""

    accuracy: float
    correct: int
    total: int
    rtol: float
    worst_relative_error: float = 0.0

    @property
    def passed(self) -> bool:
        return self.accuracy >= DEFAULT_PASS_FRACTION

    def format(self) -> str:
        return (
            f"accuracy {self.accuracy:.4f} ({self.correct}/{self.total} rows within "
            f"rtol={self.rtol:g}), worst relative error {self.worst_relative_error:.3e}"
        )


def relative_row_errors(
    actual: list[list[float]],
    reference: list[list[float]],
    floor: float = DEFAULT_FLOOR,
) -> list[float]:
    """Per-row maximum relative error against the reference.

    The max is taken *within* a row so that a row counts as correct only if all of it is
    correct; the tolerance for scattered badness is then expressed once, as the fraction
    of rows allowed to fail, rather than smeared across every element.
    """
    if len(actual) != len(reference):
        raise ValueError(
            f"shape mismatch: {len(actual)} rows against {len(reference)}; comparing "
            f"different shapes is a bug in the harness, not a kernel failure"
        )

    errors: list[float] = []
    for row_a, row_r in zip(actual, reference, strict=True):
        if len(row_a) != len(row_r):
            raise ValueError(f"row width mismatch: {len(row_a)} against {len(row_r)}")
        worst = 0.0
        for a, r in zip(row_a, row_r, strict=True):
            denom = abs(r) if abs(r) > floor else floor
            worst = max(worst, abs(a - r) / denom)
        errors.append(worst)
    return errors


def accuracy_from_errors(errors: list[float], rtol: float = DEFAULT_RTOL) -> AccuracyResult:
    """Turn per-row errors into the fraction that passed."""
    total = len(errors)
    if total == 0:
        # An empty comparison is not a perfect one. Returning 1.0 here would let a
        # harness that produced no output report a passing kernel.
        return AccuracyResult(accuracy=0.0, correct=0, total=0, rtol=rtol)
    correct = sum(1 for e in errors if e < rtol)
    return AccuracyResult(
        accuracy=correct / total,
        correct=correct,
        total=total,
        rtol=rtol,
        worst_relative_error=max(errors),
    )


def compare_against_reference(
    actual: list[list[float]],
    reference: list[list[float]],
    rtol: float = DEFAULT_RTOL,
    floor: float = DEFAULT_FLOOR,
) -> AccuracyResult:
    """Full L1-for-E3 check: per-row relative error, then the fraction within tolerance."""
    return accuracy_from_errors(relative_row_errors(actual, reference, floor), rtol)


def compare_tensors(actual, reference, rtol: float = DEFAULT_RTOL, floor: float = DEFAULT_FLOOR):
    """Same comparison for torch tensors, vectorised.

    torch is imported here rather than at module scope: Orbit's analysis path runs in
    CPU-only CI without it (§15.3), and a top-level import would make this module
    unimportable there for no reason.
    """
    import torch

    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} against {tuple(reference.shape)}")

    # Both promoted to fp32 before differencing, so the comparison is not itself
    # performed in the low precision under test.
    a = actual.to(torch.float32)
    r = reference.to(torch.float32)
    denom = r.abs().clamp_min(floor)
    rel = ((a - r).abs() / denom).flatten(start_dim=1).max(dim=-1).values

    total = int(rel.numel())
    if total == 0:
        return AccuracyResult(accuracy=0.0, correct=0, total=0, rtol=rtol)
    correct = int((rel < rtol).sum().item())
    return AccuracyResult(
        accuracy=correct / total,
        correct=correct,
        total=total,
        rtol=rtol,
        worst_relative_error=float(rel.max().item()),
    )


# ---------------------------------------------------------------------------
# §19.7 Task-level gate
# ---------------------------------------------------------------------------


class TaskVerdict(StrEnum):
    KEEP = "KEEP"
    REVERT = "REVERT"
    # The eval did not produce a usable score. Not a pass and not a failure of the
    # candidate — a failure to measure, which needs saying rather than resolving.
    UNAVAILABLE = "UNAVAILABLE"


# How far a task score may fall before a candidate is reverted.
DEFAULT_DEGRADATION = 0.05

# The absolute floor a candidate must clear regardless of the baseline.
#
# This is non-zero for a specific reason, recorded by Hyperloom as a production scar: a
# gate expressed only as "did not degrade by more than X" degenerates when the baseline
# itself is low, and at a floor of 0.0 it becomes `score > 0`. A real run of theirs KEPT
# a candidate scoring 0.00076 against a 0.906 baseline — 0.08% of it — as correct. A
# model answering essentially nothing passed a correctness gate.
DEFAULT_FLOOR_SCORE = 0.5


@dataclass
class TaskAccuracyResult:
    verdict: TaskVerdict
    baseline: float | None
    candidate: float | None
    degradation: float | None
    reason: str

    @property
    def kept(self) -> bool:
        return self.verdict is TaskVerdict.KEEP

    def format(self) -> str:
        if self.baseline is None or self.candidate is None:
            return f"[{self.verdict.value}] {self.reason}"
        return (
            f"[{self.verdict.value}] task score {self.baseline:.4f} -> {self.candidate:.4f} "
            f"({self.degradation:+.4f}) — {self.reason}"
        )


def task_accuracy_gate(
    baseline: float | None,
    candidate: float | None,
    max_degradation: float = DEFAULT_DEGRADATION,
    floor: float = DEFAULT_FLOOR_SCORE,
) -> TaskAccuracyResult:
    """Decide whether a candidate may be kept on task-level accuracy (§19.7).

    A kernel that is numerically within tolerance can still leave the served model
    broken — a wrong dtype on a rarely-hit path, a cache written but never read, a
    dispatch that silently falls back. Numerics on one kernel do not cover that, which
    is why this gate is separate from the ladder's L1 rather than folded into it.

    Two conditions, and the second is what stops the first from degenerating:

    1. The score must not fall more than `max_degradation` below the baseline.
    2. The score must clear `floor` outright, whatever the baseline was.
    """
    if baseline is None or candidate is None:
        return TaskAccuracyResult(
            verdict=TaskVerdict.UNAVAILABLE,
            baseline=baseline,
            candidate=candidate,
            degradation=None,
            reason=(
                "no task score available; correctness at the model level was not "
                "established, which is different from having been established as fine"
            ),
        )

    degradation = baseline - candidate

    if candidate < floor:
        return TaskAccuracyResult(
            verdict=TaskVerdict.REVERT,
            baseline=baseline,
            candidate=candidate,
            degradation=degradation,
            reason=(
                f"score {candidate:.4f} is below the absolute floor {floor:g}: the model "
                f"is answering close to nothing, whatever the baseline was"
            ),
        )

    if degradation > max_degradation:
        return TaskAccuracyResult(
            verdict=TaskVerdict.REVERT,
            baseline=baseline,
            candidate=candidate,
            degradation=degradation,
            reason=(
                f"dropped {degradation:.4f}, past the {max_degradation:g} allowance; a "
                f"speedup that costs this much accuracy is a different model, not a "
                f"faster one"
            ),
        )

    return TaskAccuracyResult(
        verdict=TaskVerdict.KEEP,
        baseline=baseline,
        candidate=candidate,
        degradation=degradation,
        reason=f"within the {max_degradation:g} allowance and above the {floor:g} floor",
    )
