"""
How accepted gains compose into a reported total (plan §17.6).

§17 is rigorous about measuring one change and said nothing about combining several.
The obvious move — add the accepted percentages — is wrong three times over, and wrong
in the direction that flatters the tool.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.compare.cumulative import (
    CumulativeResult,
    GainMethod,
    StackEntry,
    accumulate,
)


class TestTheHeadlineIsMeasured:
    def test_cumulative_gain_comes_from_the_final_measurement(self):
        """Not from the entries. 100 -> 131 tok/s is +31%, whatever the parts claim."""
        result = accumulate(
            100.0,
            [
                StackEntry("kernel-a", 12.0, throughput_after=112.0),
                StackEntry("kernel-b", 8.0, throughput_after=131.0),
            ],
        )
        assert result.validated_gain_percent == pytest.approx(31.0)

    def test_an_unmeasured_stack_has_no_cumulative_result(self):
        """A plausible number nobody took is worse than an explicit gap."""
        result = accumulate(
            100.0,
            [StackEntry("kernel-a", 12.0), StackEntry("kernel-b", 8.0)],
        )
        assert result.validated_gain_percent is None
        assert "NOT ESTABLISHED" in result.format()

    def test_the_summed_figure_never_appears_as_the_result(self):
        """It is computed only so the report can show what it is not claiming."""
        result = accumulate(
            100.0,
            [StackEntry("a", 12.0), StackEntry("b", 8.0)],
        )
        assert result.naive_sum_percent == pytest.approx(20.0)
        assert result.validated_gain_percent is None
        rendered = result.format()
        assert "20.00%" not in rendered.split("NOT ESTABLISHED")[0]

    def test_the_last_measured_step_stands_in_for_a_missing_final(self):
        """Still a measurement of a real stack — just not of the final one."""
        result = accumulate(
            100.0,
            [
                StackEntry("a", 12.0, throughput_after=112.0),
                StackEntry("b", 8.0),
            ],
        )
        assert result.validated_gain_percent == pytest.approx(12.0)
        assert not result.chain_continuous


class TestPercentagesDoNotAdd:
    def test_compounding_differs_from_summing(self):
        """+12% then +8% is +20.96%, not +20%. Small at two steps, not at eight."""
        result = accumulate(100.0, [StackEntry("a", 12.0), StackEntry("b", 8.0)])
        assert result.naive_sum_percent == pytest.approx(20.0)
        assert result.compounded_percent == pytest.approx(20.96)

    def test_the_gap_widens_with_the_number_of_steps(self):
        entries = [StackEntry(f"k{i}", 10.0) for i in range(8)]
        result = accumulate(100.0, entries)
        assert result.naive_sum_percent == pytest.approx(80.0)
        assert result.compounded_percent > 114.0

    def test_a_regression_in_the_stack_compounds_too(self):
        result = accumulate(100.0, [StackEntry("a", 20.0), StackEntry("b", -10.0)])
        assert result.compounded_percent == pytest.approx(8.0)


class TestDriftIsReportedNotDistributed:
    def test_parts_claiming_more_than_the_whole_shows_as_negative_drift(self):
        """Overlapping wins: two kernels on one critical path are partly one win."""
        result = accumulate(
            100.0,
            [
                StackEntry("a", 20.0, throughput_after=120.0),
                StackEntry("b", 20.0, throughput_after=131.0),
            ],
        )
        assert result.validated_gain_percent == pytest.approx(31.0)
        assert result.drift_percent == pytest.approx(31.0 - 44.0)

    def test_a_large_disagreement_is_called_out_as_a_finding(self):
        result = accumulate(
            100.0,
            [
                StackEntry("a", 25.0, throughput_after=125.0),
                StackEntry("b", 25.0, throughput_after=130.0),
            ],
        )
        assert "is a finding" in result.format()

    def test_a_small_disagreement_is_reported_without_alarm(self):
        result = accumulate(
            100.0,
            [StackEntry("a", 10.0, throughput_after=110.0)],
        )
        assert "unattributed drift" in result.format()
        assert "is a finding" not in result.format()

    def test_drift_is_unavailable_when_the_total_is(self):
        result = accumulate(100.0, [StackEntry("a", 10.0)])
        assert result.drift_percent is None


class TestGainMethodIsHonest:
    def test_a_remeasured_entry_is_measured(self):
        assert StackEntry("a", 10.0, throughput_after=110.0).gain_method is GainMethod.MEASURED

    def test_an_entry_accepted_but_never_remeasured_is_local_only(self):
        """Its local delta is real; its contribution to the stack is not established."""
        assert StackEntry("a", 10.0).gain_method is GainMethod.LOCAL_ONLY

    def test_an_entry_with_no_figure_at_all_is_missing_not_zero(self):
        """Recorded so it cannot silently vanish from the accounting."""
        assert StackEntry("a", 0.0).gain_method is GainMethod.MISSING

    def test_the_chain_is_continuous_only_when_every_step_was_remeasured(self):
        both = accumulate(
            100.0,
            [
                StackEntry("a", 10.0, throughput_after=110.0),
                StackEntry("b", 10.0, throughput_after=121.0),
            ],
        )
        assert both.chain_continuous

        one = accumulate(
            100.0,
            [StackEntry("a", 10.0, throughput_after=110.0), StackEntry("b", 10.0)],
        )
        assert not one.chain_continuous
        assert "discontinuous" in one.format()

    def test_an_empty_stack_is_not_continuous(self):
        """Nothing measured is not the same as everything measured."""
        assert not accumulate(100.0, []).chain_continuous


class TestDegenerateInputs:
    def test_a_zero_baseline_yields_no_gain_rather_than_dividing_by_zero(self):
        result = CumulativeResult(baseline_throughput=0.0, final_throughput=50.0)
        assert result.validated_gain_percent is None

    def test_an_empty_stack_still_reports_a_measured_total(self):
        """A run that changed nothing and measured drift should say so, not crash."""
        result = accumulate(100.0, [], final_throughput=99.0)
        assert result.validated_gain_percent == pytest.approx(-1.0)
