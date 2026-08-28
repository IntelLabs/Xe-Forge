"""
Correctness for a kernel that never left the framework (plan §19.6, §19.7).

§19's L1 compares an extracted bundle against a captured reference. At E3 there is no
standalone artifact, so the ladder's most-used rung had nothing behind it. The fix is
the shape AMD's Hyperloom uses: import the framework's kernel, compute a reference by
hand in higher precision, and score the fraction of rows within tolerance.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.compare.accuracy import (
    AccuracyResult,
    TaskVerdict,
    accuracy_from_errors,
    compare_against_reference,
    relative_row_errors,
    task_accuracy_gate,
)


class TestFractionWithinTolerance:
    """The metric is the fraction of rows that pass, not allclose and not a mean."""

    def test_an_exact_match_scores_one(self):
        rows = [[1.0, 2.0], [3.0, 4.0]]
        assert compare_against_reference(rows, rows).accuracy == 1.0

    def test_one_bad_row_does_not_fail_the_kernel(self):
        """`allclose` is a max over everything, so a single bad value fails everything.

        On a bf16 reduction over thousands of elements, a scattered bad value is normal.
        """
        actual = [[1.0], [2.0], [3.0], [99.0]]
        reference = [[1.0], [2.0], [3.0], [4.0]]
        result = compare_against_reference(actual, reference)
        assert result.accuracy == 0.75
        assert result.correct == 3 and result.total == 4

    def test_a_broadly_wrong_kernel_cannot_pass(self):
        """The failure mode a mean error hides: everything slightly-but-really wrong."""
        actual = [[1.5], [2.5], [3.5], [4.5]]
        reference = [[1.0], [2.0], [3.0], [4.0]]
        assert compare_against_reference(actual, reference).accuracy == 0.0

    def test_a_row_is_correct_only_if_all_of_it_is(self):
        """The max is taken within a row, so one bad element condemns that row alone."""
        actual = [[1.0, 99.0], [3.0, 4.0]]
        reference = [[1.0, 2.0], [3.0, 4.0]]
        assert compare_against_reference(actual, reference).accuracy == 0.5

    def test_the_pass_threshold_is_a_property_of_the_result(self):
        good = AccuracyResult(accuracy=0.995, correct=995, total=1000, rtol=1e-2)
        bad = AccuracyResult(accuracy=0.90, correct=900, total=1000, rtol=1e-2)
        assert good.passed and not bad.passed

    def test_the_worst_error_is_reported_alongside_the_fraction(self):
        """A 99% pass rate with a 10^6 outlier is a different situation from a clean one."""
        result = compare_against_reference([[1.0], [1e6]], [[1.0], [1.0]])
        assert result.worst_relative_error > 1e5


class TestNearZeroReferences:
    def test_relative_error_against_zero_is_floored_not_infinite(self):
        """Unbounded relative error against ~0 is a property of the reference."""
        errors = relative_row_errors([[1e-6]], [[0.0]])
        assert errors[0] == pytest.approx(1e-6 / 1e-3)

    def test_a_genuinely_wrong_near_zero_value_still_fails(self):
        """The floor must not become a way for large absolute errors to pass."""
        assert compare_against_reference([[1.0]], [[0.0]]).accuracy == 0.0


class TestDegenerateInputs:
    def test_an_empty_comparison_is_not_a_perfect_one(self):
        """Returning 1.0 would let a harness that produced nothing report success."""
        result = accuracy_from_errors([])
        assert result.accuracy == 0.0
        assert result.total == 0
        assert not result.passed

    def test_a_shape_mismatch_is_a_harness_bug_not_a_kernel_failure(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compare_against_reference([[1.0], [2.0]], [[1.0]])

    def test_a_row_width_mismatch_raises(self):
        with pytest.raises(ValueError, match="row width mismatch"):
            compare_against_reference([[1.0, 2.0]], [[1.0]])

    def test_tolerance_is_strict_at_the_boundary(self):
        """rtol is an exclusive bound, matching the reference implementation."""
        errors = [0.01]
        assert accuracy_from_errors(errors, rtol=0.01).accuracy == 0.0
        assert accuracy_from_errors(errors, rtol=0.011).accuracy == 1.0


class TestTaskAccuracyGate:
    """Numerics on one kernel do not prove the served model is intact (§19.7)."""

    def test_an_unchanged_score_is_kept(self):
        assert task_accuracy_gate(0.906, 0.906).verdict is TaskVerdict.KEEP

    def test_a_small_drop_is_within_the_allowance(self):
        assert task_accuracy_gate(0.906, 0.880).verdict is TaskVerdict.KEEP

    def test_a_large_drop_is_reverted(self):
        result = task_accuracy_gate(0.906, 0.700)
        assert result.verdict is TaskVerdict.REVERT
        assert "past the 0.05 allowance" in result.reason

    def test_an_improvement_is_kept(self):
        result = task_accuracy_gate(0.906, 0.920)
        assert result.kept
        assert result.degradation < 0

    def test_the_absolute_floor_catches_what_the_delta_rule_cannot(self):
        """Hyperloom's recorded scar: a run KEPT gsm8k=0.00076 against a 0.906 baseline.

        A gate expressed only as "did not degrade by more than X" degenerates when the
        baseline is itself low; with a floor of 0.0 it becomes `score > 0`, and a model
        answering essentially nothing passes a correctness gate.
        """
        result = task_accuracy_gate(0.00080, 0.00076)
        assert result.verdict is TaskVerdict.REVERT
        assert "absolute floor" in result.reason

    def test_the_floor_applies_even_when_the_delta_rule_would_pass(self):
        result = task_accuracy_gate(0.52, 0.49)
        assert result.degradation < 0.05  # the delta rule alone would keep it
        assert result.verdict is TaskVerdict.REVERT

    def test_a_missing_score_is_unavailable_not_a_pass(self):
        """Not established is different from established as fine."""
        result = task_accuracy_gate(0.906, None)
        assert result.verdict is TaskVerdict.UNAVAILABLE
        assert not result.kept

    def test_a_missing_baseline_is_also_unavailable(self):
        assert task_accuracy_gate(None, 0.9).verdict is TaskVerdict.UNAVAILABLE

    def test_the_verdict_renders_with_its_numbers(self):
        assert "0.9060 -> 0.7000" in task_accuracy_gate(0.906, 0.700).format()


class TestKnowledgeUnits:
    """Pin the units of the numbers Orbit hands to the optimizer.

    `amdahl_ceiling` returns a percentage; `gpu_time_share` is a fraction. Mixing them
    produced "making it infinitely fast improves end-to-end time by at most 3774.87%"
    for a kernel holding 93% of GPU time — impossible on its face, and invisible to a
    test that only checked the entry loaded.
    """

    def _kernel(self, share=0.9317, ceiling=37.75):
        from xe_forge.orbit.models import KernelRecord, Provider

        return KernelRecord(
            id="k0",
            runtime_name="gemm_kernel",
            provider=Provider.ONEDNN,
            calls=6305,
            total_time_us=1.5e6,
            avg_time_us=238.0,
            gpu_time_share=share,
            max_e2e_gain=ceiling,
        )

    def test_the_ceiling_is_reported_as_the_percentage_it_already_is(self):
        from xe_forge.orbit.knowledge import facts_for_kernel

        text = facts_for_kernel(self._kernel())[0].description
        assert "37.75%" in text
        assert "3774" not in text

    def test_the_share_is_scaled_because_it_is_a_fraction(self):
        from xe_forge.orbit.knowledge import facts_for_kernel

        assert "93.17% of GPU time" in facts_for_kernel(self._kernel())[0].description

    def test_a_ceiling_can_never_exceed_a_hundred_percent(self):
        """The invariant the bug violated. No kernel can more than eliminate itself."""
        from xe_forge.orbit.knowledge import facts_for_kernel

        for share in (0.01, 0.5, 0.93, 1.0):
            kernel = self._kernel(share=share, ceiling=share * 100)
            description = facts_for_kernel(kernel)[0].description
            reported = float(description.split("at most ")[1].split("%")[0])
            assert reported <= 100.0, description
